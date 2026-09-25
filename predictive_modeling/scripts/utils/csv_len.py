"""Count physical lines in a CSV without loading it into memory."""

import sys


def count_lines(path: str, chunk_size: int = 8 * 1024 * 1024) -> int:
    """Return the number of newline-delimited records in a file."""
    newline_count = 0
    last_byte = b""

    with open(path, "rb", buffering=chunk_size) as file:
        while chunk := file.read(chunk_size):
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]

    if last_byte and last_byte != b"\n":
        newline_count += 1

    return newline_count


def main() -> None:
    """Run the line counter from the command line."""
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <csv_path>")

    print(count_lines(sys.argv[1]))


if __name__ == "__main__":
    main()
