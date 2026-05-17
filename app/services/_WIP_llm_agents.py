# Логика - пришел запрос.
# Есть агент для поиска и агент для ответа
# 1. Получаю тип Search - если Auto - отправляю в агента по поиску, он вызывает tool: classify_query -
#    выдается QueryItem (type, точнее)
#    Если Query уже задана юзером - отправляю напрямую дальше этот тип. Query внутри один раз переводится на англ.
# 2. Query <<text>> идет в tool: extract_filters, и для нее анализируется семантика запроса - насколько релевантные фильтра можно извлечь
#    Выплевывается SearchFilters.
# 3. Query <<text>> отправляется на перефразирование и цикл поисков с контекстом (как сейчас), но с добавлением статичных фильтров и типа поиска
# 4. Когда тип result = Answer, то перенаправляем контекст на агента ответчика
# 5. Агент ответчик держит коммуникацию с юзером, анализирует насколько ответ репрезентативен еще раз. Дает конкретный сниппет песни, или 
#    задает уточняющие вопросы
#    Также он может отправлять новую базовую query юзера (см. 2+ пункты, тип query уже задан на конкретный чат.)

# SearchAgent: 
#   Базовый пайлплайн: классификация запроса, если AUTO  --> Перефразирование + tool.extract_filters (опционально) -->  
#                      анализ контекста и tool.search_db или tool.search_web если необходимо
#     tools:
#       1. classify_query, вход: query, вывод: QueryItem
#   2 + 3. rephrase_query, вход: QueryItem + предыдущие Query, вывод: RephrasedQuery. 
#   2 + 3. extract_filters, вход: raw SearchFilters, вывод: SearchFilters 
#       4. search_db, вход: RephrasedQuery + SearchFilters | None
#       5. search_web ....

# ChatAgent:
#    На основании запроса юзера может запускать SearchAgent (и др.), анализирует результат финально и выдает причесанный результат
#    Держит весь контекст диалога в памяти




# ======================== НА БУДУЩЕЕ ========================

# ResearchAgent:
#    Нужен, когда юзер запросил конкретное исследование - о чем альбом ..., какие факты есть про него, найди похожие альбомы на этот.
#    Нельзя дать все 30 текстов песен в контекст - слишком много. Надо каждый текст песни сжать до 1 предложения, 
#    и потом запустить финального ответчика
#    По звуку - использовать дополненное описание песен из классификации по CLAP (когда будет) для каждой песни еще.

filter_query_prompt = """
You are a music search planner. Your job is to analyze the user's query and prepare a search plan.

AVAILABLE FILTER KEYS:
- Artist: performer name
- Album: album name  
- Genre: music genre
- year_range: decade range (e.g. "1990-1999", "2000-2009")

INPUTS:
<user_query>{query}</user_query>
<previous_queries>{previous_queries}</previous_queries>

<resolved_filters>{resolved_filters}</resolved_filters>
<search_filter_query>{search_filter_query}</search_filter_query>

RULES:
1. If the user explicitly mentions an artist, album, genre, or time period — extract it as a potential filter. Always use ENGLISH as a filter query.
2. If <resolved_filters> is empty AND filters are needed → set action="request_filter" to look up valid DB values.
3. If <resolved_filters> is NOT empty → select the best matching filter value from it (fuzzy match allowed, e.g. "канье уест" → "Kanye West"). If no match — set filters=null.
4. You may only request filters ONCE per session. If <search_filter_query> already has content — skip request_filter.
5. Generate 2–3 search queries in English (3–10 words each), ordered from most to least specific.

OUTPUT FORMAT:
{{
  "action": "request_filter" | "search",
  "filters": {{
    "Artist": "..." | null,
    "Album": "..." | null,
    "Genre": "..." | null,
    "year_range": "YYYY-YYYY" | null
  }} | null,
  "filter_lookup": {{
    "Artist": "raw user input to resolve" | null,
    "Album": "..." | null,
    "Genre": "..." | null,
    "year_range": "YYYY-YYYY" | null
  }} | null,
  "queries": [{{"query": "..."}}],
  "search_mode": "CONSERVATIVE" | "AGGRESSIVE"
}}

NOTES:
- filter_lookup is only used when action="request_filter". It contains the raw unresolved user terms to look up in DB.
- filters contains only resolved, confirmed values from <resolved_filters> or null.
- search_mode: use CONSERVATIVE on first attempt, AGGRESSIVE if previous_queries is not empty.
""".strip()

score_answer_propmt = """
You are a music search assistant. Evaluate search results and answer the user.

INPUTS:
<user_query>{query}</user_query>
<context>{context}</context>
<previous_queries>{previous_queries}</previous_queries>
<active_filters>{active_filters}</active_filters>
<attempt>{attempt_number}</attempt>

TASK:
- Evaluate <context> for a match to <user_query>.
- If active_filters are set, ensure the result satisfies them.
- If a confident match is found → action="answer".
- If no match and attempt < 3 → action="search" with new queries (AGGRESSIVE mode).
- If attempt >= 3 → action="final_answer" (best guess or admit failure).

SEARCH MODES:
- CONSERVATIVE (attempt 1): close to user's literal words
- AGGRESSIVE (attempt 2+): use imagery, metaphors, synonyms, related themes

OUTPUT FORMAT:
{{
  "action": "answer" | "search" | "final_answer",
  "confidence": "high" | "medium" | "low",
  "song": "Title" | null,
  "artist": "Artist" | null,
  "filters": {{active filters, pass-through unchanged}} | null,
  "queries": [{{"query": "..."}}] | null,
  "message": "Conversational reply to user"
}}

CONSTRAINTS:
1. ONLY use <context> for answers. Never use internal knowledge.
2. message is ALWAYS present and human-friendly.
3. On final_answer: give best guess even if confidence is low; explain uncertainty in message.
""".strip()


from pydantic_ai import Agent


planner_agent = Agent(
    model,
    result_type=PlannerOutput,  # action: request_filter | search
    system_prompt=PLANNER_PROMPT,
)

# Агент 2 — Scorer & Answerer (бывший DEVELOPER_PROMPT)
scorer_agent = Agent(
    model,
    result_type=LLMResponse,  # SearchAction | AnswerAction
    system_prompt=SCORER_PROMPT,
)

audio_agent = Agent(
    model,
    result_type=AudioAnswer,  # {"message": str}
    system_prompt=AUDIO_ANSWER_PROMPT,
)

# В endpoint:
if effective_mode == "audio":
    # CLAP rephrase через planner_agent tool
    # search через audio_agent tool
    # ответ через audio_agent.run()



@router.post("/")
async def chat(req: ChatRequest, request: Request) -> dict:
    deps = SearchDeps(...)

    # Шаг 1: Planner (фильтры + тип запроса)
    plan = await planner_agent.run(req.message, deps=deps)

    # Шаг 1.5: если нужны фильтры из БД — resolve и перезапуск
    if plan.data.action == "request_filter":
        resolved = await resolve_filters(plan.data.filter_lookup, deps)
        plan = await planner_agent.run(
            req.message, deps=deps,
            message_history=[*plan.new_messages(), resolved]
        )

    # Шаг 2: роутинг по типу
    if plan.data.effective_mode == "audio":
        result = await audio_agent.run(req.message, deps=deps)
    else:
        result = await scorer_agent.run(req.message, deps=deps)

    return format_response(result)