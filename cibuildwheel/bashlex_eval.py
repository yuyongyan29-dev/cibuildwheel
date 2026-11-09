from __future__ import annotations

__lazy_modules__ = {"bashlex", "subprocess"}

import dataclasses
import subprocess

import bashlex

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Mapping,
        Sequence,
    )

    # a function that takes a command and the environment, and returns the result
    EnvironmentExecutor = Callable[[list[str], dict[str, str]], str]


def local_environment_executor(command: Sequence[str], env: Mapping[str, str]) -> str:
    return subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, check=True).stdout


@dataclasses.dataclass(frozen=True, kw_only=True)
class NodeExecutionContext:
    environment: dict[str, str]
    input: str
    executor: EnvironmentExecutor


def evaluate(
    value: str, environment: Mapping[str, str], executor: EnvironmentExecutor | None = None
) -> str:
    if not value:
        # empty string evaluates to empty string
        # (but trips up bashlex)
        return ""

    command_node = bashlex.parsesingle(value)

    if len(command_node.parts) != 1:
        msg = f"{value!r} has too many parts"
        raise ValueError(msg)

    value_word_node = command_node.parts[0]

    return evaluate_node(
        value_word_node,
        context=NodeExecutionContext(
            environment=dict(environment),
            input=value,
            executor=executor or local_environment_executor,
        ),
    )


def evaluate_node(node: bashlex.ast.node, context: NodeExecutionContext) -> str:
    if node.kind == "word":
        return evaluate_word_node(node, context=context)
    elif node.kind == "commandsubstitution":
        node_result = evaluate_command_node(node.command, context=context)
        # bash removes training newlines in command substitution
        return node_result.rstrip()
    elif node.kind == "parameter":
        return evaluate_parameter_node(node, context=context)
    else:
        msg = f"Unsupported bash construct: {node.kind!r}"
        raise ValueError(msg)


def evaluate_word_node(node: bashlex.ast.node, context: NodeExecutionContext) -> str:
    # bashlex stores the quote-removed text on the word, but part positions
    # index the raw source. Replacing by a textual search hits the leftmost
    # copy of that text, so a literal "\$X" before a real "$X" is expanded
    # and the real expansion is left untouched.
    offsets = _expansion_offsets(node, context.input)
    if offsets is None or not _offsets_match_source(node, context.input, offsets):
        return _substitute_parts_leftmost(node, context)

    value: str = node.word
    delta = 0
    for part, (start, end) in zip(node.parts, offsets, strict=True):
        part_value = evaluate_node(part, context=context)
        at = start + delta
        until = end + delta
        value = value[:at] + part_value + value[until:]
        delta += len(part_value) - (end - start)

    return value


def _expansion_offsets(node: bashlex.ast.node, source: str) -> list[tuple[int, int]] | None:
    """Locate each expansion inside the quote-removed word.

    Walks the raw word the way ``bashlex`` builds ``node.word``: backslash
    escapes drop the slash and keep the next character, and quote characters
    are removed (a single quote is kept only when the whole token is
    double-quoted, which is how bashlex treats it). Expansion parts are
    copied verbatim, so their source offsets land on a known span of the word.

    Returns None if those positions can't be aligned — bashlex records them
    against the token after dropping backslash-newline continuations, so the
    offsets no longer point at the original source.
    """
    if not node.parts:
        return []

    word_start = node.pos[0]
    word_end = node.pos[1]
    if word_start < 0 or word_end > len(source) or word_end < word_start:
        return None

    raw = source[word_start:word_end]
    # Sticky for the whole token: bashlex sets this from the first character,
    # it does not toggle as quotes open and close.
    double_quoted = raw.startswith('"')

    part_at: dict[int, int] = {}
    for index, part in enumerate(node.parts):
        rel_start = part.pos[0] - word_start
        rel_end = part.pos[1] - word_start
        if rel_start < 0 or rel_end > len(raw) or rel_end < rel_start or rel_start in part_at:
            return None
        part_at[rel_start] = index

    spans: list[tuple[int, int] | None] = [None] * len(node.parts)
    pieces: list[str] = []
    word_index = 0
    source_index = 0

    while source_index < len(raw):
        if any(start < source_index for start in part_at):
            return None

        part_index = part_at.pop(source_index, None)
        if part_index is not None:
            rel_end = node.parts[part_index].pos[1] - word_start
            chunk = raw[source_index:rel_end]
            spans[part_index] = (word_index, word_index + len(chunk))
            pieces.append(chunk)
            word_index += len(chunk)
            source_index = rel_end
            continue

        char = raw[source_index]
        if char == "\\":
            if source_index + 1 < len(raw):
                pieces.append(raw[source_index + 1])
                word_index += 1
            source_index += 2
        elif char == '"':
            source_index += 1
        elif char == "'":
            if double_quoted:
                pieces.append(char)
                word_index += 1
            source_index += 1
        elif char == "`" and source_index + 1 < len(raw) and raw[source_index + 1] == "`":
            pieces.append("``")
            word_index += 2
            source_index += 2
        else:
            pieces.append(char)
            word_index += 1
            source_index += 1

    if part_at or "".join(pieces) != node.word:
        return None

    located: list[tuple[int, int]] = []
    for span in spans:
        if span is None:
            return None
        located.append(span)
    return located


def _offsets_match_source(
    node: bashlex.ast.node, source: str, offsets: list[tuple[int, int]]
) -> bool:
    word: str = node.word
    for part, (start, end) in zip(node.parts, offsets, strict=True):
        if word[start:end] != source[part.pos[0] : part.pos[1]]:
            return False
    return True


def _substitute_parts_leftmost(node: bashlex.ast.node, context: NodeExecutionContext) -> str:
    value: str = node.word

    for part in node.parts:
        part_string = context.input[part.pos[0] : part.pos[1]]
        part_value = evaluate_node(part, context=context)

        if part_string not in value:
            msg = (
                f"bash parse failed. part {part_string!r} not found in {value!r}. "
                f"Word was {node.word!r}. Full input was {context.input!r}"
            )
            raise RuntimeError(msg)

        value = value.replace(part_string, part_value, 1)

    return value


def evaluate_command_node(node: bashlex.ast.node, context: NodeExecutionContext) -> str:
    if any(n.kind == "operator" for n in node.parts):
        return evaluate_nodes_as_compound_command(node.parts, context=context)
    else:
        return evaluate_nodes_as_simple_command(node.parts, context=context)


def evaluate_nodes_as_compound_command(
    nodes: Sequence[bashlex.ast.node], context: NodeExecutionContext
) -> str:
    # bashlex doesn't support any operators besides ';' inside command
    # substitutions, so we only need to handle that case. We do so assuming
    # that `set -o errexit` is on, because it's easier to code!

    result = ""
    for node in nodes:
        if node.kind == "command":
            result += evaluate_command_node(node, context=context)
        elif node.kind == "operator":
            if node.op != ";":
                msg = f"Unsupported bash operator: {node.op!r}"
                raise ValueError(msg)
        else:
            msg = f"Unsupported bash node in compound command: {node.kind!r}"
            raise ValueError(msg)

    return result


def evaluate_nodes_as_simple_command(
    nodes: Iterable[bashlex.ast.node], context: NodeExecutionContext
) -> str:
    command = [evaluate_node(part, context=context) for part in nodes]
    return context.executor(command, context.environment)


def evaluate_parameter_node(node: bashlex.ast.node, context: NodeExecutionContext) -> str:
    return context.environment.get(node.value, "")
