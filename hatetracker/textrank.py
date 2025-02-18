import csv
import os
import pathlib
import warnings
from typing import Dict, Generator, Iterable, List, Optional

import fire
import pytextrank
import spacy
import tqdm

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

PhraseList = List[Dict[str, str | float | int]]


def get_phrases(text: Iterable[str], limit: Optional[int] = None) -> List[PhraseList]:
    """Extract key phrases from a list of text strings using SpaCy's TextRank.

    Args:
        text (Iterable[str]):
            An iterable of text documents to process.
        limit (Optional[int]):
            The maximum number of phrases to extract per document, will extract
            all phrases if set to None

    Returns:
        List[PhraseList]:
            A list of lists, each containing dictionaries with phrase text,
            rank, and count.
    """
    result = list()
    for doc in nlp.pipe(text):
        curr_phrases = list()
        for phrase in doc._.phrases:
            curr_phrases.append(
                {"text": phrase.text, "rank": phrase.rank, "count": phrase.count}
            )

            if limit is not None and len(curr_phrases) >= limit:
                break

        result.append(curr_phrases)

    return result


def batch_files(
    iterable: List[str], batch_size: int
) -> Generator[List[str], None, None]:
    """Generate consecutive batches from a list of file paths.

    Args:
        iterable (List[str]): List of file paths to process.
        batch_size (int): The number of files per batch.

    Yields:
        Generator[List[str], None, None]:
            Generator yielding batches of file paths.
    """
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def batched_file_process_iter(
    file_paths: Iterable[os.PathLike],
    batch_size: Optional[int] = None,
    limit: Optional[int] = None,
) -> Generator[PhraseList, None, None]:
    """Process files in batches and yield their key phrases.

    Args:
        file_paths (Iterable[os.PathLike]):
            Paths to the text files.
        batch_size (Optional[int]):
            Number of files to process in a single batch.
        limit (Optional[int]):
            Limit the number of phrases per file.

    Yields:
        Generator[PhraseList, None, None]:
            Generator yielding lists of phrases for each file.
    """
    if batch_size is None:
        batch_size = len(file_paths)

    for batch in batch_files(file_paths, batch_size):
        texts = list()
        for file in batch:
            with open(file) as f:
                texts.append(f.read())

        for phrase_list in get_phrases(texts, limit=limit):
            yield phrase_list


def text_rank(
    input_path: os.PathLike,
    output_path: os.PathLike,
    recursive: bool = False,
    input_file_suffix: str = ".txt",
    phrase_limit: Optional[int] = None,
    file_batch_size: Optional[int] = None,
) -> None:
    """Perform TextRank keyword extraction on text files in a directory.

    This function applies the TextRank algorithm to extract key phrases from
    text files within the specified directory. The extracted phrases are saved
    in CSV format. The process can handle a single file, multiple files, or
    entire directory structures recursively.

    Args:
        input_path (os.PathLike):
            Path to the input file or directory.
        output_path (os.PathLike):
            Path where the output CSV(s) will be saved.
        recursive (bool):
            If True, process subdirectories recursively.
        input_file_suffix (str):
            The file extension to filter for processing.
        phrase_limit (Optional[int]):
            Maximum number of phrases to extract per file.
        file_batch_size (Optional[int]):
            Number of files to process in each batch.
    """
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)

    if input_path.is_file():
        input_files = [input_path]
        write_to_output_path = True
    else:
        input_files = [
            curr_file
            for curr_file in input_path.iterdir()
            if curr_file.suffix == input_file_suffix
        ]
        output_path.mkdir(exist_ok=True)
        write_to_output_path = False

    for file_path, phrases in tqdm.tqdm(
        zip(
            input_files,
            batched_file_process_iter(
                input_files, batch_size=file_batch_size, limit=phrase_limit
            ),
        ),
        desc=f"Processing file/directory {input_path}",
    ):
        if write_to_output_path:
            curr_output_path = output_path
        else:
            curr_output_path = output_path / pathlib.Path(file_path.name)
            curr_output_path = curr_output_path.with_suffix(".csv")

        if len(phrases) == 0:
            warnings.warn(
                f"No phrases found in file: {output_path}, skipping", RuntimeWarning
            )
            continue

        if curr_output_path.is_file():
            warnings.warn(
                f"File {output_path} already exists, skipping", RuntimeWarning
            )

        with open(curr_output_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=phrases[0].keys())
            writer.writeheader()
            writer.writerows(phrases)

    if not recursive or write_to_output_path:
        return

    for next_dir in input_path.iterdir():
        if not next_dir.is_dir():
            continue

        text_rank(
            next_dir,
            output_path / next_dir.name,
            recursive=True,
            input_file_suffix=input_file_suffix,
            phrase_limit=phrase_limit,
            file_batch_size=file_batch_size,
        )


def main() -> None:
    fire.Fire(text_rank)


if __name__ == "__main__":
    main()
