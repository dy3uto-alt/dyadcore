#!/usr/bin/env python3
"""Build multisession_benchmark.json from existing scenarios + new ones."""

import json


# ============================================================================
# Original 10 scenarios (reconstructed — formerly in eval_multisession.py)
# ============================================================================
OLD_SCENARIOS = [
    # ---- Scenario 1: Tech Stack 30d (project_tech, decay_ranking) ----
    {
        "name": "Tech Stack 30d (project_tech)",
        "field": "project_tech",
        "sessions": [
            (30, [
                ("user", "I use VSCode as my main editor with lots of extensions for TypeScript and React."),
                ("agent", "VSCode has excellent TypeScript support. The extension ecosystem is unmatched."),
            ]),
            (0, [
                ("user", "Switched completely to Neovim. Much faster startup and I can configure everything in Lua."),
                ("agent", "Neovim's Lua config is powerful. Many devs are making that switch for performance."),
            ]),
        ],
        "edges": [("vscode_old", "neovim_new")],
        "questions": [
            ("What code editor does the user currently use?",
             ["(a) VSCode with extensions", "(b) Neovim with Lua config", "(c) IntelliJ IDEA", "(d) Sublime Text"],
             "(b)"),
            ("What editor did the user use before switching?",
             ["(a) VSCode", "(b) Neovim", "(c) Emacs", "(d) Atom"],
             "(a)"),
            ("Why did the user switch editors?",
             ["(a) Better Git integration", "(b) Faster startup and Lua configurability",
              "(c) Better debugging support", "(d) Recommended by colleague"],
             "(b)"),
        ],
    },
    # ---- Scenario 2: 饮食偏好 120d (personal_info, decay_ranking) ----
    {
        "name": "饮食偏好 120d (personal_info)",
        "field": "personal_info",
        "sessions": [
            (120, [
                ("user", "I drink 2-3 espressos daily. Love the deep dark roast bitterness."),
                ("agent", "Espresso is a classic. The dark roast gives you that bold flavor profile."),
                ("user", "I love spicy food, especially Sichuan hotpot. The spicier the better!"),
                ("agent", "Sichuan cuisine has that signature numbing heat. It's addictive."),
            ]),
            (0, [
                ("user", "After my health checkup, switched to decaf and herbal tea. No more espresso."),
                ("agent", "Decaf and herbal teas are much gentler on your system. Good health choice."),
                ("user", "Developed stomach issues so now I prefer mild Cantonese steamed dishes and Japanese food."),
                ("agent", "Cantonese cuisine focuses on fresh ingredients and gentle cooking. Much easier on digestion."),
            ]),
        ],
        "edges": [("espresso_old", "decaf_new"), ("spicy_old", "mild_new")],
        "questions": [
            ("What does the user drink now?",
             ["(a) Espresso, 2-3 cups daily", "(b) Decaf and herbal tea", "(c) Black tea", "(d) Green smoothies"],
             "(b)"),
            ("What cuisine does the user prefer now?",
             ["(a) Sichuan hotpot", "(b) Mild Cantonese and Japanese", "(c) Korean BBQ", "(d) Italian pasta"],
             "(b)"),
            ("Why did the user change their diet?",
             ["(a) New year resolution", "(b) Health reasons — checkup and stomach issues",
              "(c) Influenced by friends", "(d) Moved to a new city"],
             "(b)"),
        ],
    },
    # ---- Scenario 3: Bug 10d (bugs_issues, decay_ranking) ----
    {
        "name": "Bug 10d (bugs_issues)",
        "field": "bugs_issues",
        "sessions": [
            (10, [
                ("user", "Critical bug: Login page has infinite redirect loop on iOS Safari browser."),
                ("agent", "iOS Safari sometimes handles redirect chains differently. Need to investigate the auth flow."),
            ]),
            (0, [
                ("user", "The iOS Safari login redirect bug is now fixed and deployed to production."),
                ("agent", "The fix was a missing state parameter in the OAuth callback. Good catch."),
            ]),
        ],
        "edges": [("bug_active_old", "bug_fixed_new")],
        "questions": [
            ("What is the current status of the iOS Safari login bug?",
             ["(a) Still active and investigating", "(b) Fixed and deployed to production",
              "(c) Cannot reproduce", "(d) Waiting for Apple to fix"],
             "(b)"),
            ("What was the original bug symptom?",
             ["(a) App crashes on launch", "(b) Infinite redirect loop on login",
              "(c) Blank white screen", "(d) Slow page loading"],
             "(b)"),
        ],
    },
    # ---- Scenario 4: Sprint 5d (meeting_notes, decay_ranking) ----
    {
        "name": "Sprint 5d (meeting_notes)",
        "field": "meeting_notes",
        "sessions": [
            (5, [
                ("user", "Sprint planning decision: Q2 focus will be on performance optimization across all services."),
                ("agent", "Performance work is long overdue. We should target p95 latency reduction."),
            ]),
            (0, [
                ("user", "Performance optimization goals achieved. Now shifting focus to reliability engineering."),
                ("agent", "Reliability is the natural next step. SLO-based alerting and chaos engineering."),
            ]),
        ],
        "edges": [("perf_old", "reliability_new")],
        "questions": [
            ("What is the current sprint focus?",
             ["(a) Performance optimization", "(b) Reliability engineering", "(c) Feature development", "(d) Security audit"],
             "(b)"),
            ("What was the previous sprint goal?",
             ["(a) Performance optimization", "(b) Reliability engineering", "(c) UI redesign", "(d) Database migration"],
             "(a)"),
        ],
    },
    # ---- Scenario 5: 3段部署演进 (project_tech, multi_stage_evolution) ----
    {
        "name": "3段部署演进 (project_tech)",
        "field": "project_tech",
        "sessions": [
            (90, [
                ("user", "Our deployment is all shell scripts. scp tarballs to servers and run setup.sh manually."),
                ("agent", "Shell scripts work for simple setups but become hard to maintain as you grow."),
            ]),
            (45, [
                ("user", "Moved from shell scripts to Docker Compose. Much more reproducible deployments now."),
                ("agent", "Docker Compose gives you environment consistency. Big improvement over shell scripts."),
            ]),
            (0, [
                ("user", "Now using Kubernetes with Helm charts. CI/CD via GitHub Actions pushes to our K8s cluster."),
                ("agent", "Kubernetes is the gold standard for container orchestration at scale."),
            ]),
        ],
        "edges": [("shell_old", "docker_mid"), ("docker_mid", "k8s_new")],
        "questions": [
            ("What is the user's CURRENT deployment method?",
             ["(a) Shell scripts with scp", "(b) Docker Compose", "(c) Kubernetes with Helm and GitHub Actions",
              "(d) AWS Lambda serverless"],
             "(c)"),
            ("What deployment method did the user use FIRST?",
             ["(a) Shell scripts", "(b) Docker Compose", "(c) Kubernetes", "(d) Terraform"],
             "(a)"),
            ("What did the user migrate FROM to get to Kubernetes?",
             ["(a) Shell scripts directly", "(b) Docker Compose", "(c) Heroku", "(d) Ansible"],
             "(b)"),
        ],
    },
    # ---- Scenario 6: Anchor 冲突 (personal_info, anchor_conflict) ----
    {
        "name": "Anchor 冲突 (personal_info)",
        "field": "personal_info",
        "sessions": [
            (180, [
                ("user", "I'm a strict vegetarian. No meat, no fish — it's an ethical choice I've held for years."),
                ("agent", "That's a strong commitment. Vegetarianism for ethical reasons is admirable."),
                ("user", "I drink Coke with every meal. It's my guilty pleasure."),
                ("agent", "Coke is a classic but that's a lot of sugar. Maybe try sparkling water sometimes."),
            ]),
            (0, [
                ("user", "Started eating fish occasionally. I'm pescatarian now — needed more protein options."),
                ("agent", "Pescatarian gives you omega-3s and lean protein. A practical middle ground."),
                ("user", "Switched from Coke to just water and unsweetened tea. Feel much better."),
                ("agent", "Cutting out sugary drinks is one of the best health moves you can make."),
            ]),
        ],
        "edges": [("vegetarian_old", "pescatarian_new"), ("coke_old", "water_new")],
        "questions": [
            ("What is the user's current diet preference?",
             ["(a) Strict vegetarian", "(b) Pescatarian — eats fish occasionally",
              "(c) Full omnivore", "(d) Vegan"],
             "(b)"),
            ("What was the user's previous diet?",
             ["(a) Strict vegetarian", "(b) Pescatarian", "(c) Keto", "(d) Mediterranean"],
             "(a)"),
            ("What does the user drink with meals now?",
             ["(a) Coke", "(b) Water and unsweetened tea", "(c) Orange juice", "(d) Beer"],
             "(b)"),
        ],
    },
    # ---- Scenario 7: 跨场域隔离 (field_isolation) ----
    {
        "name": "跨场域隔离 — MySQL 检索隔离",
        "field": "project_tech",
        "sessions": [
            (60, [
                ("user", "Our search feature uses MySQL LIKE queries for full-text search on product descriptions."),
                ("agent", "MySQL LIKE works for simple search but doesn't scale well. Consider full-text indexes."),
            ]),
            (60, [
                ("user", "I collect MySQL-related memorabilia. Got a vintage MySQL dolphin plushie from a 2006 conference."),
                ("agent", "That's a rare collector's item! The MySQL dolphin mascot has gone through several redesigns."),
            ]),
            (0, [
                ("user", "Search is now on Elasticsearch. We dropped MySQL LIKE queries entirely."),
                ("agent", "Elasticsearch is purpose-built for search. Much better relevance and performance."),
            ]),
        ],
        "edges": [("mysql_old", "elasticsearch_new")],
        "questions": [
            ("What search technology does the project currently use?",
             ["(a) MySQL LIKE queries", "(b) Elasticsearch", "(c) PostgreSQL full-text", "(d) Algolia"],
             "(b)"),
            ("What personal item does the user own?",
             ["(a) PostgreSQL elephant statue", "(b) MySQL dolphin plushie", "(c) Linux penguin figure", "(d) Docker whale toy"],
             "(b)"),
        ],
    },
    # ---- Scenario 8: 部分更新 (project_tech, partial_update) ----
    {
        "name": "部分更新 — 仅后端 (project_tech)",
        "field": "project_tech",
        "sessions": [
            (60, [
                ("user", "Our stack is Express.js backend with React frontend. REST API between them."),
                ("agent", "Express + React is a battle-tested combo. Simple and effective for most apps."),
            ]),
            (0, [
                ("user", "We rewrote the backend in FastAPI (Python). Frontend is still React — no changes there."),
                ("agent", "FastAPI gives you async performance and auto-generated OpenAPI docs. Frontend stays the same."),
            ]),
        ],
        "edges": [("express_old", "fastapi_new")],
        "questions": [
            ("What backend framework does the user currently use?",
             ["(a) Express.js", "(b) FastAPI", "(c) Django", "(d) Ruby on Rails"],
             "(b)"),
            ("What frontend framework does the user use?",
             ["(a) React (unchanged)", "(b) Vue 3", "(c) Angular", "(d) Svelte"],
             "(a)"),
        ],
    },
    # ---- Scenario 9: 隐式漂移 (personal_info, implicit_drift) ----
    {
        "name": "隐式漂移 — 无显式切换 (personal_info)",
        "field": "personal_info",
        "sessions": [
            (365, [
                ("user", "I watch Netflix every night after work. Binged all the top thrillers this year."),
                ("agent", "Netflix has an incredible thriller catalog. Hard to stop once you start bingeing."),
                ("user", "Chopin's nocturnes are my favorite music. I listen while reading."),
                ("agent", "Chopin is timeless. The nocturnes create a perfect contemplative atmosphere."),
            ]),
            (0, [
                ("user", "Went to a classical music concert at the symphony hall last weekend. The live experience was transcendent."),
                ("agent", "Live classical performances have a depth that recordings can't fully capture."),
                ("user", "Been getting into jazz lately. Miles Davis and John Coltrane are opening up a whole new world."),
                ("agent", "Jazz is an incredible journey. Davis and Coltrane are the perfect gateway artists."),
            ]),
        ],
        "edges": [("netflix_old", "concert_new"), ("classical_old", "jazz_new")],
        "questions": [
            ("What entertainment does the user currently prefer?",
             ["(a) Netflix thrillers at home", "(b) Live classical concerts and jazz music",
              "(c) Video games", "(d) Sports events"],
             "(b)"),
            ("What music is the user currently into?",
             ["(a) Chopin nocturnes", "(b) Jazz — Miles Davis and Coltrane", "(c) Rock music", "(d) Electronic dance"],
             "(b)"),
            ("What was the user's previous evening activity?",
             ["(a) Watching Netflix", "(b) Going to concerts", "(c) Playing sports", "(d) Cooking"],
             "(a)"),
        ],
    },
    # ---- Scenario 10: 超长衰减 365d (personal_info, extreme_decay) ----
    {
        "name": "超长衰减 365d (personal_info)",
        "field": "personal_info",
        "sessions": [
            (365, [
                ("user", "I'm a die-hard fiction reader. Thrillers and mysteries are my absolute favorites."),
                ("agent", "Thrillers keep you turning pages. The suspense craft in good mysteries is unmatched."),
            ]),
            (0, [
                ("user", "Now I mainly read nonfiction — behavioral economics and cognitive science. Much more practical."),
                ("agent", "Behavioral economics explains so much about how we actually make decisions vs how we think we do."),
            ]),
        ],
        "edges": [("fiction_old", "nonfiction_new")],
        "questions": [
            ("What genre does the user currently read?",
             ["(a) Fiction thrillers and mysteries", "(b) Nonfiction — behavioral economics and cognitive science",
              "(c) Science fiction", "(d) Historical fiction"],
             "(b)"),
            ("What was the user's previous reading preference?",
             ["(a) Fiction thrillers", "(b) Nonfiction economics", "(c) Poetry", "(d) Biography"],
             "(a)"),
        ],
    },
]


def _convert_edges(edges):
    """Convert legacy label-tuple edges to keyword-based dict edges."""
    if not edges:
        return []
    # Map problematic label suffixes to proper search keywords
    KEYWORD_FIX = {
        "githubactions": "GitHub Actions",
        "rediscluster": "Redis Cluster",
        "redis old": "Redis single",
        "mysql old": "MySQL LIKE",
        "docker old": "Docker Compose",
        "bug active old": "Critical bug",
        "bug fixed new": "fixed",
        "perf old": "performance optimization",
        "reliability new": "reliability engineering",
        "espresso old": "espresso",
        "decaf new": "decaf",
        "spicy old": "Sichuan",
        "mild new": "Cantonese",
        "coke old": "Coke",
        "water new": "water",
        "shell old": "shell scripts",
        "docker mid": "Docker",
        "k8s new": "Kubernetes",
        "vegetarian old": "vegetarian",
        "pescatarian new": "pescatarian",
        "fiction old": "thrillers",
        "nonfiction new": "behavioral economics",
        "netflix old": "Netflix",
        "concert new": "classical music",
        "classical old": "Chopin",
        "jazz new": "jazz",
        "express old": "Express.js",
        "fastapi new": "FastAPI",
    }
    result = []
    for edge in edges:
        if isinstance(edge, dict):
            result.append(edge)
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            old_raw = edge[0]
            new_raw = edge[1]
            # Look up in KEYWORD_FIX using space-normalized form
            old_space = old_raw.replace("_", " ").strip()
            new_space = new_raw.replace("_", " ").strip()
            old_kw = KEYWORD_FIX.get(old_space, old_raw)
            new_kw = KEYWORD_FIX.get(new_space, new_raw)
            result.append({"old_keyword": old_kw, "new_keyword": new_kw})
    return result


scenarios = []

for sc in OLD_SCENARIOS:
    js = {
        "name": sc["name"],
        "field": sc["field"],
        "description": "",
        "sessions": [
            {
                "age_days": age_days,
                "messages": [{"role": r, "content": c} for r, c in msgs],
            }
            for age_days, msgs in sc["sessions"]
        ],
        "edges": _convert_edges(sc.get("edges", [])),
        "questions": [
            {"question": q, "options": opts, "correct": corr, "tests": "current_preference"}
            for q, opts, corr in sc["questions"]
        ],
    }

    name_lower = sc["name"].lower()
    if "3段" in name_lower or "deploy" in name_lower:
        js["description"] = "3-stage evolution: shell→Docker→K8s"
        js["capability"] = "multi_stage_evolution"
    elif "tech stack" in name_lower:
        js["description"] = "Tests time decay ranking of old vs new technical preferences across 30-day gap"
        js["capability"] = "decay_ranking"
    elif "饮食" in name_lower or "diet" in name_lower:
        js["description"] = "Tests long-gap (120d) preference revision on personal health/diet choices"
        js["capability"] = "decay_ranking"
    elif "bug" in name_lower:
        js["description"] = "Tests short-gap (10d) bug lifecycle tracking"
        js["capability"] = "decay_ranking"
    elif "sprint" in name_lower:
        js["description"] = "Tests very-short-gap (5d) meeting note goal evolution"
        js["capability"] = "decay_ranking"
    elif "anchor" in name_lower:
        js["description"] = "Anchored old preference vs new pescatarian preference"
        js["capability"] = "anchor_conflict"
    elif "跨场域" in name_lower or "cross" in name_lower:
        js["description"] = "Same keyword in different fields — tests field isolation"
        js["capability"] = "field_isolation"
    elif "部分" in name_lower or "partial" in name_lower:
        js["description"] = "Partial stack update: only backend changes"
        js["capability"] = "partial_update"
    elif "隐式" in name_lower or "implicit" in name_lower:
        js["description"] = "Preference drift without explicit switch statements"
        js["capability"] = "implicit_drift"
    elif "超长" in name_lower or "long" in name_lower:
        js["description"] = "365-day decay test: old memory beyond half-life"
        js["capability"] = "extreme_decay"

    for q in js["questions"]:
        ql = q["question"].lower()
        if "before" in ql or "previous" in ql or "original" in ql:
            q["tests"] = "old_preference_recall"
        elif "why" in ql or "reason" in ql:
            q["tests"] = "causal_reasoning"
        else:
            q["tests"] = "current_preference"

    scenarios.append(js)

# ---- NEW SCENARIOS ----

new_scenarios = [
    {
        "name": "噪声会话干扰 (project_tech, 14d + noise)",
        "field": "project_tech",
        "description": "Key preference info embedded in noisy conversations about unrelated topics",
        "capability": "noise_resilience",
        "sessions": [
            {
                "age_days": 14,
                "messages": [
                    {"role": "user", "content": "Our main database is PostgreSQL with JSONB columns for flexible schemas."},
                    {"role": "agent", "content": "PG JSONB is great for semi-structured data with full queryability."},
                    {"role": "user", "content": "Hey, did you watch the Champions League final last night? What a match!"},
                    {"role": "agent", "content": "I heard it was thrilling! Football brings people together like nothing else."},
                    {"role": "user", "content": "For monitoring we use Grafana dashboards hooked up to Prometheus metrics."},
                    {"role": "agent", "content": "Grafana + Prometheus is the standard open-source observability stack."},
                    {"role": "user", "content": "I need to buy a birthday gift for my nephew. Any ideas for a 10-year-old?"},
                    {"role": "agent", "content": "LEGO sets or a beginner robotics kit are always hits with that age group."},
                    {"role": "user", "content": "We handle auth with JWT tokens stored in HTTP-only cookies, refresh every 15min."},
                    {"role": "agent", "content": "JWT in HTTP-only cookies is secure. Short refresh windows limit exposure."},
                ],
            },
            {
                "age_days": 0,
                "messages": [
                    {"role": "user", "content": "Switched from PostgreSQL JSONB to a dedicated MongoDB for flexible schema parts."},
                    {"role": "agent", "content": "MongoDB is purpose-built for document stores. Good call for that use case."},
                    {"role": "user", "content": "Can you recommend a good sushi place in downtown? Taking a client out Friday."},
                    {"role": "agent", "content": "Sushi Nakazawa is highly rated for business dinners. Omakase menu impresses."},
                    {"role": "user", "content": "We replaced Prometheus + Grafana with Datadog for better integrated monitoring."},
                    {"role": "agent", "content": "Datadog costs more but the integrated APM + logs + metrics in one pane is worth it."},
                    {"role": "user", "content": "Auth is now session-based with Redis-backed session store, dropped JWT entirely."},
                    {"role": "agent", "content": "Server-side sessions give you instant revocation. Important for security-sensitive apps."},
                ],
            },
        ],
        "edges": [
            {"old_keyword": "PostgreSQL", "new_keyword": "MongoDB"},
            {"old_keyword": "Grafana", "new_keyword": "Datadog"},
            {"old_keyword": "JWT", "new_keyword": "session-based"},
        ],
        "questions": [
            {
                "question": "What database does the user currently use for flexible schemas?",
                "options": ["(a) PostgreSQL with JSONB", "(b) MongoDB", "(c) MySQL", "(d) Cassandra"],
                "correct": "(b)",
                "tests": "current_preference",
            },
            {
                "question": "What monitoring solution does the user use now?",
                "options": ["(a) Prometheus + Grafana", "(b) Datadog", "(c) New Relic", "(d) Splunk"],
                "correct": "(b)",
                "tests": "current_preference",
            },
            {
                "question": "How does the user handle authentication now?",
                "options": ["(a) JWT tokens in HTTP-only cookies", "(b) Session-based with Redis store",
                            "(c) OAuth2 with Google", "(d) Basic auth with passwords"],
                "correct": "(b)",
                "tests": "current_preference",
            },
        ],
    },
    {
        "name": "多维度同步演化 (personal_info, 60d, 3 tracks)",
        "field": "personal_info",
        "description": "Three independent preferences (diet, exercise, media) evolve simultaneously",
        "capability": "multi_dimension_evolution",
        "sessions": [
            {
                "age_days": 60,
                "messages": [
                    {"role": "user", "content": "I run 5K three times a week, training for a half-marathon later this year."},
                    {"role": "agent", "content": "Consistent 5K runs build a great aerobic base for half-marathon training."},
                    {"role": "user", "content": "I mostly watch sci-fi series. Just finished The Expanse, amazing show."},
                    {"role": "agent", "content": "The Expanse sets a high bar for sci-fi. Hard to top that world-building."},
                    {"role": "user", "content": "For breakfast I usually have cereal with milk and a banana. Quick and easy."},
                    {"role": "agent", "content": "Simple breakfasts work. Carbs for energy and potassium from the banana."},
                ],
            },
            {
                "age_days": 0,
                "messages": [
                    {"role": "user", "content": "Switched from running to swimming. Better for my knees and full-body workout."},
                    {"role": "agent", "content": "Swimming is zero-impact and excellent cardio. Great choice for joint health."},
                    {"role": "user", "content": "Got into historical documentaries lately. Finished a great series on ancient Rome."},
                    {"role": "agent", "content": "Historical documentaries can be just as gripping as fiction when well-produced."},
                    {"role": "user", "content": "Changed breakfast to Greek yogurt with granola and berries. More protein, less sugar."},
                    {"role": "agent", "content": "Greek yogurt has double the protein of regular yogurt. Berries add antioxidants."},
                ],
            },
        ],
        "edges": [
            {"old_keyword": "running", "new_keyword": "swimming"},
            {"old_keyword": "sci-fi", "new_keyword": "documentaries"},
            {"old_keyword": "cereal", "new_keyword": "yogurt"},
        ],
        "questions": [
            {
                "question": "What exercise does the user currently do?",
                "options": ["(a) Running 5K three times a week", "(b) Swimming", "(c) Weight lifting", "(d) Yoga"],
                "correct": "(b)",
                "tests": "current_preference",
            },
            {
                "question": "What genre does the user currently watch?",
                "options": ["(a) Sci-fi series", "(b) Historical documentaries", "(c) Reality TV", "(d) Comedy shows"],
                "correct": "(b)",
                "tests": "current_preference",
            },
            {
                "question": "What does the user eat for breakfast now?",
                "options": ["(a) Cereal with milk and banana", "(b) Greek yogurt with granola and berries",
                            "(c) Eggs and toast", "(d) Smoothie bowl"],
                "correct": "(b)",
                "tests": "current_preference",
            },
        ],
    },
    {
        "name": "半衰期边界 bugs 6d vs 8d (half=7d)",
        "field": "bugs_issues",
        "description": "Two bugs at 6d and 8d — straddling the 7d half-life boundary",
        "capability": "decay_boundary",
        "sessions": [
            {
                "age_days": 8,
                "messages": [
                    {"role": "user", "content": "Bug B-201: Dashboard charts not rendering on Firefox ESR, SVG transform issue."},
                    {"role": "agent", "content": "SVG transform compatibility varies across browsers. Firefox ESR is conservative."},
                ],
            },
            {
                "age_days": 6,
                "messages": [
                    {"role": "user", "content": "Bug B-202: Checkout page times out for orders with more than 15 line items."},
                    {"role": "agent", "content": "Timeout on large orders suggests a database query optimization problem."},
                ],
            },
            {
                "age_days": 0,
                "messages": [
                    {"role": "user", "content": "Both bugs B-201 (Firefox SVG) and B-202 (checkout timeout) are now fixed and deployed."},
                    {"role": "agent", "content": "SVG fix was a CSS workaround. Checkout fix was a query index improvement."},
                ],
            },
        ],
        "edges": [
            {"old_keyword": "B-201", "new_keyword": "fixed"},
            {"old_keyword": "B-202", "new_keyword": "fixed"},
        ],
        "questions": [
            {
                "question": "Which bug was reported EARLIER?",
                "options": ["(a) B-201 - Firefox SVG rendering", "(b) B-202 - Checkout timeout",
                            "(c) Both at the same time", "(d) Cannot determine"],
                "correct": "(a)",
                "tests": "temporal_ordering",
            },
            {
                "question": "What is the status of both bugs?",
                "options": ["(a) Both still open", "(b) Both fixed and deployed",
                            "(c) Only B-201 is fixed", "(d) Only B-202 is fixed"],
                "correct": "(b)",
                "tests": "current_preference",
            },
        ],
    },
    {
        "name": "半衰期边界 project_tech 55d vs 65d (half=60d)",
        "field": "project_tech",
        "description": "Two migrations straddling the 60d half-life: 55d (within) vs 65d (past)",
        "capability": "decay_boundary",
        "sessions": [
            {
                "age_days": 65,
                "messages": [
                    {"role": "user", "content": "We used to host everything on Heroku. Simple but gets expensive at scale."},
                    {"role": "agent", "content": "Heroku is great for MVPs and small teams. The managed experience saves DevOps time."},
                ],
            },
            {
                "age_days": 55,
                "messages": [
                    {"role": "user", "content": "Moved from Heroku to AWS EC2 with Terraform for infrastructure-as-code."},
                    {"role": "agent", "content": "Terraform + EC2 gives you control and cost efficiency at scale."},
                ],
            },
            {
                "age_days": 0,
                "messages": [
                    {"role": "user", "content": "Now using AWS ECS Fargate instead of raw EC2. Serverless containers are simpler."},
                    {"role": "agent", "content": "Fargate eliminates the EC2 instance management overhead. True pay-per-use."},
                ],
            },
        ],
        "edges": [
            {"old_keyword": "Heroku", "new_keyword": "EC2"},
            {"old_keyword": "EC2", "new_keyword": "Fargate"},
        ],
        "questions": [
            {
                "question": "What infrastructure does the user CURRENTLY use?",
                "options": ["(a) Heroku", "(b) AWS EC2 with Terraform", "(c) AWS ECS Fargate", "(d) Google Cloud Run"],
                "correct": "(c)",
                "tests": "current_preference",
            },
            {
                "question": "What did the user use BEFORE EC2?",
                "options": ["(a) Heroku", "(b) AWS ECS Fargate", "(c) DigitalOcean", "(d) Azure"],
                "correct": "(a)",
                "tests": "old_preference_recall",
            },
            {
                "question": "What infrastructure did the user migrate FROM to get to Fargate?",
                "options": ["(a) Heroku directly", "(b) AWS EC2 with Terraform", "(c) Google Cloud Run", "(d) Bare metal servers"],
                "correct": "(b)",
                "tests": "causal_reasoning",
            },
        ],
    },
    {
        "name": "快速连续修正 (personal_info, 3d to 0d)",
        "field": "personal_info",
        "description": "Preference revises twice in 3 days. Only the latest counts.",
        "capability": "rapid_revision",
        "sessions": [
            {
                "age_days": 3,
                "messages": [
                    {"role": "user", "content": "I want to learn guitar. Been listening to a lot of flamenco music lately."},
                    {"role": "agent", "content": "Flamenco guitar is technically demanding but incredibly expressive."},
                ],
            },
            {
                "age_days": 1,
                "messages": [
                    {"role": "user", "content": "Actually no, I think piano would be better. More versatile and easier to learn theory on."},
                    {"role": "agent", "content": "Piano is the best instrument for learning music theory. All notes laid out linearly."},
                ],
            },
            {
                "age_days": 0,
                "messages": [
                    {"role": "user", "content": "Changed my mind again. Going with drums. I just want to hit things and make rhythm!"},
                    {"role": "agent", "content": "Drums are pure rhythmic joy. Sometimes the simplest motivation is the best one."},
                ],
            },
        ],
        "edges": [
            {"old_keyword": "guitar", "new_keyword": "piano"},
            {"old_keyword": "piano", "new_keyword": "drums"},
        ],
        "questions": [
            {
                "question": "What instrument does the user want to learn NOW?",
                "options": ["(a) Guitar - flamenco style", "(b) Piano - for music theory",
                            "(c) Drums - for rhythm", "(d) Violin"],
                "correct": "(c)",
                "tests": "current_preference",
            },
            {
                "question": "What was the user's FIRST choice of instrument?",
                "options": ["(a) Guitar (flamenco)", "(b) Piano", "(c) Drums", "(d) Violin"],
                "correct": "(a)",
                "tests": "old_preference_recall",
            },
            {
                "question": "Why did the user settle on drums?",
                "options": ["(a) Cheapest instrument", "(b) Wants to hit things and make rhythm",
                            "(c) Recommended by a friend", "(d) Easiest to carry"],
                "correct": "(b)",
                "tests": "causal_reasoning",
            },
        ],
    },
]

scenarios.extend(new_scenarios)

benchmark = {
    "name": "DyadCore Multi-Session Preference Revision Benchmark",
    "version": "1.0",
    "description": "Evaluates memory systems on multi-session preference tracking, temporal reasoning, and knowledge update across field-specific decay half-lives.",
    "capabilities": sorted(set(
        s.get("capability", "unknown") for s in scenarios
    )),
    "scenarios": scenarios,
}

with open("multisession_benchmark.json", "w", encoding="utf-8") as f:
    json.dump(benchmark, f, ensure_ascii=False, indent=2)

total_qs = sum(len(s["questions"]) for s in scenarios)
by_cap = {}
for s in scenarios:
    cap = s.get("capability", "unknown")
    by_cap[cap] = by_cap.get(cap, 0) + len(s["questions"])
by_tests = {}
for s in scenarios:
    for q in s["questions"]:
        t = q.get("tests", "unknown")
        by_tests[t] = by_tests.get(t, 0) + 1

print(f"Scenarios: {len(scenarios)}")
print(f"Total questions: {total_qs}")
print(f"\nBy capability:")
for cap, count in sorted(by_cap.items(), key=lambda x: -x[1]):
    print(f"  {cap}: {count}")
print(f"\nBy test type:")
for t, count in sorted(by_tests.items(), key=lambda x: -x[1]):
    print(f"  {t}: {count}")
print("\nSaved: multisession_benchmark.json")
