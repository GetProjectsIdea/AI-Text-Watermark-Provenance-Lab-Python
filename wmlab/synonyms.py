"""A small hand-made synonym table for word-substitution attacks (offline, no WordNet needed)."""
from __future__ import annotations

_GROUPS = """
good fine nice decent excellent
bad poor awful terrible
great large big huge vast
small little tiny slight
happy glad pleased cheerful joyful
sad unhappy sorrowful gloomy
said remarked replied answered observed
asked inquired enquired
think believe suppose imagine
know understand realise
see notice observe perceive
look glance gaze stare
walk stroll wander
go proceed move travel
come arrive approach
begin start commence
end finish conclude close
quick fast rapid swift
quickly rapidly swiftly hastily
slow gradual leisurely
slowly gradually
very really extremely exceedingly
often frequently repeatedly
always constantly invariably
never seldom rarely
perhaps maybe possibly
certainly surely definitely undoubtedly
house home residence dwelling
room chamber apartment
man gentleman fellow person
woman lady
friend companion acquaintance
child kid youngster
father dad parent
mother mum parent
young youthful juvenile
old aged elderly ancient
beautiful lovely pretty handsome attractive
ugly plain unattractive
rich wealthy affluent
afraid frightened scared fearful
angry cross furious annoyed
strange odd peculiar curious unusual
dark dim gloomy shadowy
light bright luminous
cold chilly cool frosty
hot warm heated
night evening dusk
morning dawn daybreak
day daytime
road street path way lane
city town
letter note message
money cash funds
work labour toil job
help aid assist support
want wish desire
need require
like enjoy fancy
love adore cherish
hate detest loathe dislike
give hand offer provide
take seize grab
keep retain hold
get obtain acquire receive
make create produce
find discover locate
tell inform
show display reveal
hear listen
feel sense
try attempt endeavour
leave depart quit
stay remain linger
return come_back
answer reply response
question query enquiry
idea notion thought concept
reason cause motive
moment instant minute second
time occasion
place spot location position
part portion piece share
whole entire complete
many numerous several
few some
important significant major
true correct accurate
false untrue wrong incorrect
easy simple effortless
hard difficult tough
clear plain obvious evident
sure certain confident
quiet silent calm still
loud noisy
strong powerful mighty
weak feeble frail
kind gentle caring
cruel harsh brutal
clever smart intelligent bright
foolish silly stupid
wonderful marvellous amazing
terrible dreadful horrible
indeed truly really
however nevertheless nonetheless
also too besides
about regarding concerning
almost nearly practically
enough sufficient adequate
again once_more
therefore thus hence consequently
because since as
while whilst
until till
upon on
""".strip()

SYNONYMS: dict[str, list[str]] = {}
for line in _GROUPS.splitlines():
    words = [w.replace("_", " ") for w in line.split()]
    for w in words:
        if " " in w:
            continue
        alts = [x for x in words if x != w]
        SYNONYMS.setdefault(w, [])
        SYNONYMS[w].extend(a for a in alts if a not in SYNONYMS[w])


def synonyms(word: str) -> list[str]:
    return SYNONYMS.get(word.lower(), [])
