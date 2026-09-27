# RAG Mini-Agent — tanulási projekt leírása (2026-09-16)

## Cél

Hands-on tapasztalat RAG + Agent (function-calling / tool-választás) + opcionálisan MCP témában, valós haszonnal: Károly PDF dokumentumainak (pl. szerződések, céges papírok, egyéb iratok) rendszerezése és kérdezhetővé tétele (jelenleg nem tudja, melyik PDF mit tartalmaz).

Kontextus: ez a projekt a freeCodeCamp AI Agent tutorial melletti saját gyakorlás, a 2026-09-16-i LeadFlowAutomation-interjú (AI Agentic Developer for B2B Platforms) felkészülésének részeként indult, de nem az interjúra készül élesben bemutatásra — tanulási cél, saját tempóban.

## Miért agent, nem sima RAG-pipeline

A klasszikus RAG (kérdés → embedding → retrieval → prompt → válasz) egy fix, determinisztikus pipeline, nem agent. Az agent-jelleget az adja, hogy a modell két különálló tool közül **maga választ** a user szándéka alapján:

- `add_document(file_path)` — új dokumentum (PDF vagy Markdown) hozzáadása a tudásbázishoz
- `query_knowledge_base(question)` — kérdés megválaszolása a már betárolt dokumentumok alapján

A modell function-calling módban dönti el, melyiket hívja meg egy adott user-üzenetre.

## Architektúra

- **Tárolás:** Postgres + pgvector (Supabase, amit Károly már ismer napi szinten)
- **PDF szövegkinyerés:** `pypdf` vagy `pdfplumber` (utóbbi jobb, ha a PDF-ekben táblázatok is vannak)
- **Darabolás (chunking):** ~300-500 token/darab, kis átfedéssel — ld. `claude/rag-gyorstalpalo.md`
- **Embedding:** egy embedding API (pl. OpenAI `text-embedding-3` vagy amit Károly elérhetőnek talál)
- **Metaadat minden darabhoz:** forrás-fájlnév, oldalszám — hogy a válasz visszamutasson, melyik PDF-ből jött

## Ismert korlátok (tudatosan nem ma megoldandó)

- Szkennelt (kép-alapú) PDF-eknél a szövegkinyerés üres/szemét eredményt adhat — OCR (pl. Tesseract) kellene hozzá, ez külön feladat, nem blokkolja a mai gyakorlást.
- Adatvédelem: a feldolgozott dokumentumok (pl. személyes adatok, céges/szerződéses tartalom) kimennek egy külső LLM/embedding API-hoz feldolgozásra. Személyes tanulóprojektként elfogadható, tudatosan vállalt döntés — ha éles, más felet is érintő rendszer épülne erre, ezt előre tisztázni kell az érintett féllel/munkáltatóval.
- ~~Chunk-szintű token-túllépés csak figyelmeztetést kap, nincs automatikus korrekció~~ — **megoldva (2026-09-17)**: `CHUNK_OVERFLOW_STRATEGY` (`.env`) Strategy pattern szerint (`ingestion/chunker.py`: `WarnOverflowStrategy` a régi becslés-alapú figyelmeztetés, `SplitOverflowStrategy` az új korrekció). `split` módban a driver valódi tokenizálójával (`EmbeddingDriver.count_tokens()`, `LocalSentenceTransformerDriver`-en a nyers HF tokenizer, `model.tokenize()` nélkül — az utóbbi már csonkolna) méri le minden chunk tényleges token-számát, és bináris kereséssel újra-darabolja, ami túllép — semmi nem vágódik le csendben. Ha az aktív driver nem tud valódi token-számot adni (pl. OpenAI), `split` visszaesik `warn`-ra, figyelmeztetéssel. Valós PDF-en tesztelve: 21 szó-alapú chunkból 4 lépte túl a 128 tokenes limitet, ezeket a korrekció 8 darabra bontotta (végeredmény 29 chunk, semmi csonkolás).
  - **Overlap-arány felvetés (még nyitott)**: ha a `CHUNK_SIZE`-ot a valós token-limithez igazítva drasztikusan csökkentenénk, a jelenlegi fix `CHUNK_OVERLAP` (abszolút szószám) értelmetlenül nagy aránnyá válhat. Külön, még megbeszélendő feladat — a `split` stratégia ezt nem oldja meg, csak a csendes csonkolást szünteti meg.
- ~~`TEST_DATABASE_URL` ugyanaz volt, mint a `DATABASE_URL`~~ — **megoldva (2026-09-17)**: bevezettük az `AGENT_ENV` mechanizmust (`config.py`) + `.env.test`-et. A `db_module`-t korábban monkeypatch-elő `tests/db/conftest.py` fixture is törölhetővé vált, mert `AGENT_ENV=test` alatt a `settings.DATABASE_URL` eleve helyesen a teszt-DB-re mutat a teljes folyamat alatt. Lásd README "AGENT_ENV and .env.test".
- ~~Oldalhatáron átfutó bekezdés két csonka chunk-ká esett szét~~ — **megoldva (2026-09-18)**: `chunk_pages()` (oldalankénti darabolás) helyett az `ingest.py` most `pdf_loader.extract_document_text()` + `chunker.chunk_document()`-et használja — a teljes dokumentum egybefűzve kerül darabolásra, nincs többé oldal-loop, ami elvágná a határon átfutó szöveget. Két új, egymástól független `.env` dimenzió: `PDF_EXTRACTION_MODE` (`flat`/`blocks`) és `CHUNKING_STRATEGY` (`word`/`langchain`, utóbbi `langchain_text_splitters.RecursiveCharacterTextSplitter`-rel bekezdés-/mondathatáron tördel). Mindkettő default-ja a régi viselkedést adja vissza (backward-compatible). `chunk_pages()` változatlanul megmaradt (tesztek, `scripts/extract_text.py`).
  - **Ismert korlát a `blocks` módban**: a bekezdés-detektálás `pdfplumber` szó-koordinátákon (y-ugrás a sortávolság medián × 1.8-szorosa felett) alapul, ami oldalanként független — oldaltörésen **nem** tud bekezdéshatárt észlelni (nincs mivel összehasonlítani a következő oldal első szavának koordinátáját). A küszöb (1.8×median) sem tökéletes minden dokumentumon — valós PDF-eken mérve (két helyi teszt-PDF, sosem commitolva — ld. "Incidens, folyamatban" lentebb) néhány azonos-bekezdésen-belüli sortávolság is a határ közelébe esett. Mivel a `blocks` mód opt-in (nem default), ez elfogadott v1 korlát, finomhangolás később, ha felmerül az igény.
  - **Oldal-metaadat pontossága**: mivel egy chunk szavai most több oldalról is jöhetnek, a `page_number` többségi szavazással (`chunk_document()`) dől el — a legtöbb szót adó oldal száma kerül a metaadatba, nem egy oldaltartomány. Egyszerű, visszafelé kompatibilis, a ritka off-by-one-oldal hiba elhanyagolható ár azért, hogy ne kelljen a prompt-template-et és a retrieval-megjelenítést egy oldaltartomány-formátumra átalakítani.
- **Markdown (`.md`/`.markdown`) fájlok beolvasása** — **megoldva (2026-09-18)**: `ingestion/extractors.py`, `Extractor` Strategy pattern (`PDFExtractor` + `MarkdownExtractor`), a felhasználó saját javaslata alapján. Egyetlen tudatos eltérés a projekt eddigi Strategy-mintájától: a kiválasztás (`get_extractor()`) nem `.env`-ből, hanem a fájlkiterjesztésből történik — a fájltípus tény, nem preferencia (ld. `AGENTS.md`). A `MarkdownExtractor` nem futtat semmilyen `blocks`-szerű koordináta-heurisztikát, mert a Markdown natívan tartalmazza a saját bekezdés-/szakasz-szerkezetét (üres sorok, `#` fejlécek) — egyszerűen beolvassa a fájlt. Mivel Markdown-nak nincsenek valódi oldalai, a `page_number` metaadat itt egy fejléc-alapú szakasz-számláló (minden `#`...`######` sor új szakaszt nyit) — ugyanaz a mező, más egység, ugyanaz a pragmatikus "ne bővítsük a metaadat-sémát" döntés, mint a fenti oldal-szavazásnál. Code-fence (```` ``` ````) védelem is bekerült, hogy egy dokumentációs fájl kódpéldájában lévő `#`-kommentár ne számítson tévesen fejlécnek. `pdf_loader.py` változatlan — az `extractors.py` csak újrahasznosítja a meglévő függvényeit.
- **Incidens, lezárva a jelen munkafában, history-takarítás még hátravan (2026-09-18)**: a `tests/data/`-ban commitolt "teszt" PDF-ek (eredetileg `App_Screen_Updates_v3.pdf`, majd `test_1.pdf`/`test_2.pdf`) valójában **valódi, érzékeny freelance ügyfél-tartalmat** hordoztak (konkrét árazás, projekt-/ügyfélnév, technikai specifikáció) — tévesen "ártalmatlan teszt-adatnak" lettek besorolva, holott a gépen elérhető PDF-ek természetükből adódóan valós dokumentumok. Soha nem lettek push-olva (a GitHub remote a takarítás előtti kommitnál állt). A fájlok törlése most már commitolva van (`feat: add Markdown document support...`), és találtunk egy valóban ártalmatlan PDF-et (`tests/data/sample.pdf`, saját MVP termékterv, ügyfél/PII nélkül — le is ellenőriztük a teljes kinyert szöveget), ami most a `.pdf` fixture (`fix: strip unmapped-glyph markers...` kommit). **Még hátravan**: `git filter-branch --index-filter` a lokális history-ból (`refs/heads/main`-re szűkítve, nem `--all` — a korábbi stash-incidens tanulsága miatt) a három érzékeny történeti névre — a jelenlegi HEAD-en már nincs bennük semmi, de a régebbi kommitokban (history) igen.
- **Code review megfigyelés — `char_start` fragilitás a `LangChainChunkingStrategy`-ben (2026-09-18)**: egy független code review assert-et javasolt a "minden split-pont szóhatáron van" feltevésre (`LangChainChunkingStrategy.split()`). **Elvetve**: egy assert egy ritka, alacsony-hatású metaadat-pontossági hibát (1-2 szónyi `page_number`-csúszás egy chunk-határnál) garantált leállássá változtatna — rosszabb, nem jobb. Helyette empirikusan igazoltuk, hogy a `keep_separator=False` (a reviewer egy másik, önálló TIP-jéből) **megszünteti** a leggyakoribb forrását (`". "` szeparátor a darab elejéhez ragadása) — a maradék, extrém ritka eset (egyetlen, a teljes chunk-budgetnél is hosszabb "szó", ami az utolsó, `""` szeparátor-fallback-et kényszeríti ki) ugyanolyan elfogadott korlát, mint a `word` stratégia `_split_oversized_text`-jénél. Részletes indoklás: `ingestion/chunker.py`'s `LangChainChunkingStrategy` docstring + a `fix: address independent code review findings...` kommit üzenete.

## Lépések (interaktívan, egyenként megbeszélve)

1. Projekt-mappa + Python venv + szükséges csomagok (`pypdf`/`pdfplumber`, embedding-klienshez való lib, Postgres-klienshez való lib)
2. pgvector tábla létrehozása a Supabase-projektben
3. PDF → szöveg kinyerés, tesztelve egy valós PDF-en
4. Darabolás + embedding + tárolás (`add_document` logika)
5. Lekérdezés: embedding + top-k retrieval + válasz-generálás (`query_knowledge_base` logika)
6. A két tool összekötése function-callinggal, hogy a modell döntsön, melyiket hívja
7. ✅ **Stretch goal, elkészült:** a két tool becsomagolva `mcp_server.py`-ban, stdio transporttal (Claude Desktop-hoz `make mcp-install`) — FastAPI/Docker (hálózaton elérhető szerver) explicit nincs scope-ban, mert az interjút nem érinti.

## Munkamódszer

Károly a saját gépén, helyi Claude Code-dal implementálja, de lépésről lépésre, tanító jelleggel megy végig rajta ebben a session-ben is (Murphy-vel), nem csak a végeredményt nézi át.
