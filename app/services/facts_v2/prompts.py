"""Prompts, v2 — the label set the validation settled on.

Conventions from app/services/assistant/prompts.py: worked examples beat rules,
the prompt asks for material rather than decisions, the output contract is
repeated at the end, English only.

One rule of our own, learned the hard way: an INSTRUCTION never contains
quotable content. v1 illustrated "drop the subject" with «Записана за одну
ночь…» inside the instruction, and that sentence came back as a fact about a
song nothing was recorded overnight about. Illustrations live in EXAMPLES,
where they are paired input→output and cannot be mistaken for material.
"""

from __future__ import annotations

SONG_LABELS = {"sample", "video", "creation", "sound", "placement", "trouble",
               "title_origin", "record", "about_artist", "other"}
ARTIST_LABELS = {"band_history", "personal", "name_origin", "award", "sound",
                 "about_song", "other"}

# Which prompt drives a multi-label fact: most specific first. The more specific
# the class, the less obvious its must-survive list — an award's "won or merely
# nominated" is easy to lose, while band_history's demands overlap everyone's.
SPECIFICITY = ["name_origin", "title_origin", "trouble", "record", "award",
               "video", "placement", "sound", "creation", "personal",
               "band_history"]


SONG_CLASSIFY = """You sort raw notes about ONE song into labels. You do not rewrite them.

The song is "{title}" by {artist}.

Two kinds of note arrive here. Some are editorial entries from a song database.
Some are fan notes written next to a single lyric line, and they look like
`Line: "..." | Note: ...`. For a fan note you judge WHAT THE NOTE CLAIMS, never
the lyric line it hangs on: a note quoting a striking line and then explaining
what it means carries no fact at all.

A note earns a real label only if it is about THIS song. A note about a
different song, or about nothing in particular, is "other".

LABELS

"sample" — the recording contains a sample or an interpolation of another
recording, or another recording sampled this one. Sound only. Lyrics that quote
another song's words are not a sample. A producer credit is not a sample.
  · "The drum break is lifted from The Winstons' 'Amen, Brother'." → sample
  · "He raps the same line Nas used on 'The World Is Yours'." → other

"video" — the music video for this song: who directed it, how and where it was
shot, what happens in it, how it was received.

"creation" — how the song came to exist: writing, recording, studio incidents,
who it was written for, why it was released the way it was, and what the people
who made it have said about making it.

"title_origin" — where the song's TITLE comes from: a person, a place, a work,
a phrase, a word's meaning. This is about the title itself, not about what the
lyrics are about.
  · "Named after the German physicist Georg Simon Ohm; an ohm is the unit of
     electrical resistance." → title_origin
  · "The song is about resisting a manipulative ex." → other

"trouble" — a lawsuit, a plagiarism claim, a ban, a withdrawal, a formal
complaint, a boycott, over THIS song.

"about_artist" — the note's subject is the artist, and this song is only where
the note happens to sit.

"sound" — how the record sounds and how that was achieved: instruments, playing
technique, arrangement, production choices, time signature, vocal treatment.

"placement" — where this recording has been heard OUTSIDE its own release:
films, television, series, video games, adverts, sports broadcasts, political
campaigns. A cover version by another artist is NOT this label.

"record" — a record or a first that this song holds: an award it won, a chart
feat stated as a record, an entry in a book of records. A plain chart position
is not this label; a record is.
  · "Its debut at #42 made 98-year-old Marjorie Grande the oldest credited
     artist ever to appear on the Hot 100." → record
  · "It reached number 4 on the Billboard Hot 100." → other

"other" — everything else, and it is the honest answer far more often than the
labels above. In particular: a retelling of what the lyrics mean, a metaphor
explained, a general reflection on life, love, money or fame, a release date or
chart position standing on its own, a cover version by another artist, a slang
or nickname gloss with no story behind it, and anything about a different song.
  · "The 'heart' is a symbol of determination, so he is saying he is still
     resilient." → other
  · "It was covered by Brenda Lee and by Hall & Oates." → other

A note may carry more than one label — "sample" and "sound" often travel
together. "other" never combines with anything.

MOVING A NOTE

When you label a note "about_artist", also say where it belongs on the artist's
own page, using the artist labels: "band_history", "personal", "name_origin",
"award", "sound", "other". Put that in `move`. Otherwise `move` is null.

EXAMPLES

Notes:
M1. The bassline is a slowed-down loop of Chic's "Good Times".
M2. Line: "I'm the king of pain" | Note: He feels that his suffering defines him, and the crown is an ironic one.
M3. Recorded in a single night in a Hollywood garage after their gear was stolen.
M4. Bono has won 22 Grammy Awards across his career.
M5. Line: "Two for no" | Note: A way of communicating used by patients with locked-in syndrome, the illness portrayed in the film The Diving Bell and the Butterfly.
M6. The title comes from the Magnolia Projects in New Orleans, the housing project where the rapper grew up.

{{"items":[
  {{"id":"M1","labels":["sample","sound"],"move":null}},
  {{"id":"M2","labels":["other"],"move":null}},
  {{"id":"M3","labels":["creation"],"move":null}},
  {{"id":"M4","labels":["about_artist"],"move":{{"scope":"artist","labels":["award"]}}}},
  {{"id":"M5","labels":["other"],"move":null}},
  {{"id":"M6","labels":["title_origin"],"move":null}}
]}}

(M5 is "other" on purpose: the note explains an image inside the lyric. It names
a real film, and that still does not make it a fact ABOUT THE SONG.)

NOTES TO SORT

{items}

Answer with STRICT JSON and nothing else — no markdown fence, no text around it.
Echo every id exactly as given. Allowed labels: "sample", "video", "creation",
"title_origin", "trouble", "about_artist", "sound", "placement", "record",
"other".
{{"items":[{{"id":"M1","labels":["..."],"move":null}}]}}"""


ARTIST_CLASSIFY = """You sort raw notes about ONE artist into labels. You do not rewrite them.

The artist is {artist}.

A note earns a real label only if it is about this artist. A note about somebody
else, or about nothing in particular, is "other".

LABELS

"about_song" — the note is really about ONE named song by this artist: how it
was made, what it samples, its video, where it has been heard. The song must be
the note's SUBJECT, not a passing mention.

"award" — a Grammy, a chart record, a hall-of-fame induction, a certification, a
national or festival prize, or a notable anti-prize.

"band_history" — how the group formed, split, reformed, changed members, moved
city, signed or left a label; the events that make up the group's story.
It is about the GROUP. A thing that happened to one person, and would still be
worth telling if the group did not exist, is "personal".

"personal" — a concrete, specific thing about the PERSON behind the music: a job
they did before music, an illness or diagnosis, an arrest or conviction, a
tattoo and how they got it, military service, a phobia, a collection, a family
tie, a near-miss that changed their life. It must carry a specific detail or a
story. A taste, a preference, a diet or a compliment is not this.
  · "He tattooed O,Z,Z,Y across his own knuckles with a needle and graphite
     while in jail for burglary." → personal
  · "He spent nine years in the Navy, where speech therapy cured his stutter,
     then wrote songs while delivering milk." → personal
  · "Cuomo has a thing for female newscasters." → other
  · "Ariana Grande is a vegan and loves animals more than most people." → other

"name_origin" — where the artist's or band's name came from, and who chose it.

"sound" — the artist's signature sound and how it is made: instruments,
production habits, arrangement tics, vocal approach, influences on their style.
It must describe THE MUSIC. A life event that later fed into a record is
"personal", not "sound".

"other" — everything else, and it is the honest answer often: opinions, tastes,
vague praise, plain discography chronology with nothing behind it, a fact about
a completely different artist, and a bare roster or release listing with no
event in it.

A note may carry more than one label. "other" never combines with anything.

MOVING A NOTE

When you label a note "about_song", also give the song's title exactly as the
note spells it, and where it belongs on that song's page, using the song labels:
"sample", "video", "creation", "title_origin", "trouble", "sound", "placement",
"record", "other". Put that in `move`. Otherwise `move` is null.

EXAMPLES

Notes:
M1. The band is named after the spiral galaxy M83.
M2. Bellamy wanted to be in a classical or jazz band and set out to learn jazz piano.
M3. "Lonely Boy" got its video by accident: they scrapped the shoot and kept an hour of Derrick Tuggle dancing.
M4. Formed in Antibes in 2001 by Anthony Gonzalez and Nicolas Fromageau; Fromageau left after the second album.
M5. At 21, after his band broke up and his girlfriend left him, he drank furniture polish in a suicide attempt.
M6. 1979- Paul Dean Guitar, vocals 1979- Doug Johnson Keyboards 1979- Mike Reno Lead vocals

{{"items":[
  {{"id":"M1","labels":["name_origin"],"move":null}},
  {{"id":"M2","labels":["other"],"move":null}},
  {{"id":"M3","labels":["about_song"],"move":{{"scope":"song","title":"Lonely Boy","labels":["video"]}}}},
  {{"id":"M4","labels":["band_history"],"move":null}},
  {{"id":"M5","labels":["personal"],"move":null}},
  {{"id":"M6","labels":["other"],"move":null}}
]}}

NOTES TO SORT

{items}

Answer with STRICT JSON and nothing else — no markdown fence, no text around it.
Echo every id exactly as given. Allowed labels: "about_song", "award",
"band_history", "personal", "name_origin", "sound", "other".
{{"items":[{{"id":"M1","labels":["..."],"move":null}}]}}"""


# ── stage 2 ──────────────────────────────────────────────────────────────────

REFINE = """You rewrite ONE fact for a music player. You do not judge it — it has
already been chosen.

CRITICAL RULE — NAMES KEEP THEIR SCRIPT. Names of PEOPLE, BANDS, SONGS, ALBUMS,
FILMS, GAMES, BRANDS and LABELS stay written the way the source writes them. A
name in Latin letters stays in Latin letters, character for character, and it
does not get declined. Rendering one in the alphabet of {lang} is the worst
error you can make here.
Countries, cities and everyday words are NOT names in this sense — translate
those normally, and never leave an ordinary English noun untranslated.

  source: Don Felder gave him guitar lessons.
  WRONG:  Дон Фелдер давал ему уроки игры на гитаре.
  RIGHT:  Don Felder давал ему уроки игры на гитаре.

  source: written with his frequent collaborator Samuel Dixon
  WRONG:  с его частым collaborator Samuel Dixon
  RIGHT:  с его постоянным соавтором Samuel Dixon

{subject}

WHAT MUST SURVIVE

{focus}

HOW TO WRITE IT

- Write in {lang}. The whole answer is in {lang}: only the names stay Latin.
- One sentence, two at the very most, and never longer than {max_chars}
  characters or longer than the fact you were given.
- Every name, number, date and place in your answer must already appear in THE
  FACT below. If the fact does not say who directed it, you do not say who
  directed it. Adding anything is inventing it.
- A detail you have no room for is dropped, never replaced with a guess.
- Do not open with a filler clause.
{name_rule}

EXAMPLES

{shots}

THE FACT

{fact}

Answer with STRICT JSON and nothing else — no markdown fence, no text around it.
The answer is written in {lang}; Latin-spelled names stay in Latin letters.
{{"text":"..."}}"""


SONG_SUBJECT = """This fact will sit next to the song it belongs to, and the
listener already sees the title above it. Do not write the title. When the
sentence has to point at the song, use an ordinary noun phrase in {lang},
declined to fit the sentence, and never inside quotation marks. When the
sentence reads perfectly well without pointing at the song at all — which is
most of the time — leave the pointer out. Do not bolt one onto a sentence whose
real subject is a video, a film, an advert or another performer."""

ARTIST_SUBJECT = """This fact will sit on the artist's own page, and the listener is
already looking at their name. You may name the artist when the sentence needs
it, but you do not have to."""


# Per-label length caps. A single global cap put 21.7% of the run over the line,
# almost all of it band_history, where the length is legitimate.
MAX_CHARS = {
    "band_history": 300, "creation": 260, "record": 260, "personal": 260,
    "trouble": 240, "award": 240, "video": 220, "placement": 220,
    "sound": 220, "name_origin": 220, "title_origin": 220,
}

FOCUS = {
    "video": """The director, where and how it was shot, the year, and the one image
or incident that makes it worth reading. Drop production trivia that carries
none of those.""",

    "creation": """Who made it, where, when, and under what circumstances — the
concrete detail that makes it a story rather than a schedule.
A quotation from the people who made it is the usual shape of these facts.
RETELL it compactly in your own words, keeping who said it and what they
claimed; do not carry the quotation across verbatim.
The ONE exception: when the quotation explains WHY THE SONG IS CALLED WHAT IT IS,
keep those words verbatim and set them as a markdown quote with `> `.""",

    "title_origin": """What the title literally comes from — the person, place, work,
phrase or word — and who chose it. If the source explains what the word itself
means, that meaning is the point and must survive.""",

    "trouble": """Who acted against whom, on what grounds, in what year, and how it
ended. An unresolved case must read as unresolved.""",

    "sound": """The instrument, technique or production choice, who played or made it,
and what it does to the record — the thing a listener could go and hear.""",

    "placement": """The exact name of the film, series, game, advert or broadcast, the
year, and what was happening on screen when it played.""",

    "record": """What the record actually is, the exact number that makes it one, the
year, and whose record it displaced if the source says. A record without its
number is not a record.""",

    "award": """The award and its level, the year, the exact category, and whether it
was won or only nominated. Never blur a nomination into a win.""",

    "band_history": """The year, the place, the people, and the reason — a lineup
change without its cause is a list, not a fact.""",

    "personal": """The concrete detail and what came of it — the specific thing that
makes this a story about a person rather than a trait. Keep the numbers, the
places and the jobs; they are what make it real.""",

    "name_origin": """The literal origin of the name and who chose it. If the members
disagree about it, that disagreement IS the fact.""",
}

SHOTS = {
    "video": """Fact: Directed by Chris Cunningham in an abandoned Los Angeles hospital over two nights in 1997; the dancers were cast from a local ballet school and told nothing about the plot.
{"text":"Клип снял Chris Cunningham за две ночи 1997 года в заброшенной больнице Лос-Анджелеса — танцовщиц взяли из местной балетной школы и не рассказали им сюжет."}

Fact: The video was shot in one take on a Steadicam. It won Video of the Year at the 2003 MTV Video Music Awards.
{"text":"Клип снят одним дублем на стедикам и взял «Видео года» на MTV Video Music Awards 2003."}""",

    "creation": """Fact: Songwriter Tom Nichols said in 1000 UK #1 Hits: "William Orbit produced it and it came out sounding fantastic. It was completely different from the demo." The song was originally intended for Kirsty Roper, but London Records executive Tracy Bennett wanted it for All Saints.
{"text":"Песню писали для Kirsty Roper, но Tracy Bennett из London Records забрал её для All Saints; продюсировал William Orbit, и, по словам автора Tom Nichols, результат вышел совсем не похожим на демо."}

Fact: Asked why it carries that title, Mercury replied: "It's about a bicycle race, and it's about nothing else. I saw the Tour de France go past the studio window and that was that."
{"text":"Название родилось буквально из окна студии — мимо шёл «Тур де Франс».\\n\\n> It's about a bicycle race, and it's about nothing else. I saw the Tour de France go past the studio window and that was that."}""",

    "title_origin": """Fact: "Ohms" is the title track of the Deftones' ninth studio album. Named after the German physicist Georg Simon Ohm, an ohm is a unit of electrical resistance between two points of a conductor.
{"text":"Название взято у немецкого физика Georg Simon Ohm: ом — единица электрического сопротивления."}

Fact: The song is titled after the infamous Magnolia Projects in New Orleans, the crime-ridden housing project that was home to a number of southern rappers.
{"text":"Название отсылает к Magnolia Projects — печально известному жилому кварталу Нового Орлеана, откуда вышло немало южных рэперов."}""",

    "trouble": """Fact: In 2015 a Los Angeles jury found that the song copied Marvin Gaye's "Got to Give It Up" and awarded Gaye's children $7.4 million, later reduced to $5.3 million on appeal.
{"text":"В 2015 году суд Лос-Анджелеса признал, что песня копирует «Got to Give It Up» Marvin Gaye, и присудил его детям 7,4 миллиона долларов — в апелляции сумму снизили до 5,3 миллиона."}

Fact: The BBC declined to add it to its playlist in 1977, and several retailers refused to stock it; the band's label never confirmed a formal ban.
{"text":"В 1977 году BBC не поставила её в ротацию, а часть магазинов отказалась держать сингл на полках — формального запрета лейбл так и не подтвердил."}""",

    "sound": """Fact: The song features an imaginative solo played exclusively on bicycle bells, unusual chord progressions and shifts in time signature from 4/4 to 6/8.
{"text":"Соло здесь сыграно целиком на велосипедных звонках, а размер по ходу переключается с 4/4 на 6/8."}

Fact: Jonny Greenwood ran the guitar through an Ondes Martenot, an early electronic instrument, which produces the wavering high line in the chorus.
{"text":"Дрожащую высокую линию в припеве даёт волны Мартено — ранний электронный инструмент, на котором играет Jonny Greenwood."}""",

    "placement": """Fact: The song was featured in the first-season Friends episode "The One Where Underdog Gets Away", playing over shots of the posters all over New York City.
{"text":"Звучит в серии «The One Where Underdog Gets Away» из первого сезона Friends — под кадры с расклеенными по Нью-Йорку плакатами."}

Fact: Rockstar licensed it for the Los Santos Rock Radio station in Grand Theft Auto V (2013).
{"text":"Rockstar лицензировала её для радиостанции Los Santos Rock Radio в Grand Theft Auto V (2013)."}""",

    "record": """Fact: The song debuted on the Billboard Hot 100 at #42, making 98-year-old Marjorie "Nonna" Grande the oldest credited artist to appear on the chart. She surpassed the previous record held by Fred Stobaugh, who at 96 was featured on "Sweet Lorraine".
{"text":"Дебют на #42 в Billboard Hot 100 сделал 98-летнюю Marjorie «Nonna» Grande самой возрастной артисткой в истории чарта — прежний рекорд держал 96-летний Fred Stobaugh."}

Fact: They held the Guinness Book of World Records title of the World's Loudest Band (117 dB) in the 1975-76 edition.
{"text":"В издании Книги рекордов Гиннесса 1975–1976 годов они значились самой громкой группой мира — 117 дБ."}""",

    "award": """Fact: Isolation (2018) was nominated for Best Urban Contemporary Album at the 61st Annual Grammy Awards but lost to The Carters' Everything Is Love.
{"text":"Альбом Isolation (2018) номинировался на «Грэмми» в категории Best Urban Contemporary Album, но уступил Everything Is Love дуэта The Carters."}

Fact: The band was inducted into the Rock and Roll Hall of Fame in 2019, their first year of eligibility.
{"text":"Группу ввели в Rock and Roll Hall of Fame в 2019 году — в первый же год, когда она получила на это право."}""",

    "band_history": """Fact: Irons and Slovak formed a group called What Is This? as a side project in the early 1980s, which contractually prevented them from recording with the Red Hot Chili Peppers, but they quickly returned to the lineup to tour.
{"text":"В начале 1980-х Irons и Slovak собрали побочный проект What Is This?, и контракт с ним не давал им записываться с Red Hot Chili Peppers — в концертный состав они вернулись быстро."}

Fact: Formed in Antibes in 2001 by Anthony Gonzalez and Nicolas Fromageau; Fromageau left after the second album and Gonzalez has run the project alone since.
{"text":"Коллектив собрали в Антибе в 2001 году Anthony Gonzalez и Nicolas Fromageau; после второго альбома Fromageau ушёл, и с тех пор проект держит один Gonzalez."}""",

    "personal": """Fact: He spent nine years in the Navy, where he had speech therapy to overcome his stuttering. After his release from the Navy, he wrote songs as he delivered milk, then worked at Ford Motor Company and IBM.
{"text":"Девять лет он прослужил во флоте, где логопед помог ему справиться с заиканием, а после службы писал песни, развозя молоко, и работал в Ford Motor Company и IBM."}

Fact: An arsonist burned his house down in 1987. Petty, along with his wife, daughter and housekeeper, were in the house at the time but escaped largely unharmed.
{"text":"В 1987 году поджигатель сжёг его дом — Petty с женой, дочерью и домработницей были внутри и выбрались почти невредимыми."}""",

    "name_origin": """Fact: The members disagree about the name. Bob Hardy: "Mainly we just liked the way it sounded. We liked the alliteration." Alex Kapranos: "His life, or at least the ending of it, was the catalyst for the complete transformation of the world and that is what we want our music to be."
{"text":"Насчёт названия участники расходятся: Bob Hardy говорит, что им просто нравилась аллитерация, а Alex Kapranos — что гибель эрцгерцога перевернула мир, и такой же музыки они хотят сами."}

Fact: The band is named after M83, a spiral galaxy in the constellation Hydra.
{"text":"Название взято у M83 — спиральной галактики в созвездии Гидры."}""",
}


SAMPLE_EXTRACT = """You extract a sampling link from ONE note about the song
"{title}" by {artist}. You do not rewrite the note.

A sampling link has a direction:
  "source" — THIS song samples or interpolates another recording;
  "usage"  — ANOTHER recording sampled or interpolated THIS song.

And a relation:
  "sample"        — the other record's actual audio is used;
  "interpolation" — the melody or part is replayed rather than lifted.

Give the other recording's performer and title exactly as the note spells them.
Latin spellings stay Latin, character for character. If the note names no other
recording — if it only says "this has a sample in it", or names a producer, or
quotes a lyric — return {{"links":[]}}.

Do not look for a year or an album. They are rarely stated and you must not
guess them.

EXAMPLES

Note: The drum break is lifted from The Winstons' "Amen, Brother".
{{"links":[{{"direction":"source","artist":"The Winstons","title":"Amen, Brother","relation":"sample"}}]}}

Note: Kanye West replayed the string line from Curtis Mayfield's "Move On Up" for this beat, and Drake later flipped this song's hook on "0 to 100".
{{"links":[{{"direction":"source","artist":"Curtis Mayfield","title":"Move On Up","relation":"interpolation"}},
          {{"direction":"usage","artist":"Drake","title":"0 to 100","relation":"sample"}}]}}

Note: Produced by Rick Rubin, who layered the drums heavily.
{{"links":[]}}

THE NOTE

{fact}

Answer with STRICT JSON and nothing else — no markdown fence, no text around it.
{{"links":[{{"direction":"source|usage","artist":"...","title":"...","relation":"sample|interpolation"}}]}}"""
