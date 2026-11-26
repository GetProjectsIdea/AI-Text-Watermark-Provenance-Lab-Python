# Bundled corpora

Public-domain texts from Project Gutenberg (via the GITenberg mirror), with the Project Gutenberg
header and licence footer removed and stored gzip-compressed:

| file | contents | used for |
|---|---|---|
| `train.txt.gz` | Jane Austen, *Pride and Prejudice*; Arthur Conan Doyle, *The Adventures of Sherlock Holmes* | training the toy trigram LM |
| `human.txt.gz` | Mary Shelley, *Frankenstein*; Charles Dickens, *A Tale of Two Cities* | held-out human text (false-positive tests, mixing attack) |
| `alice.txt.gz` | Lewis Carroll, *Alice's Adventures in Wonderland* | second human style |

These works are in the public domain in India and the United States.
