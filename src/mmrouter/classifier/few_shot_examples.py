"""Few-shot examples for LLM-based classification."""

FEW_SHOT_EXAMPLES = [
    # simple / factual
    {"prompt": "What is the speed of light?", "complexity": "simple", "category": "factual"},
    {"prompt": "How many planets are in the solar system?", "complexity": "simple", "category": "factual"},

    # simple / reasoning
    {"prompt": "Why does metal feel colder than wood at the same temperature?", "complexity": "simple", "category": "reasoning"},
    {"prompt": "What is the difference between hunger and appetite?", "complexity": "simple", "category": "reasoning"},

    # simple / creative
    {"prompt": "Write a two-line poem about a cat.", "complexity": "simple", "category": "creative"},
    {"prompt": "Make up a fun name for a bakery.", "complexity": "simple", "category": "creative"},

    # simple / code
    {"prompt": "What does the print() function do in Python?", "complexity": "simple", "category": "code"},
    {"prompt": "What is a Git branch?", "complexity": "simple", "category": "code"},

    # medium / factual
    {"prompt": "How does the stock market work?", "complexity": "medium", "category": "factual"},
    {"prompt": "Explain how nuclear reactors generate electricity.", "complexity": "medium", "category": "factual"},

    # medium / reasoning
    {"prompt": "What are the trade-offs between renting and buying a home?", "complexity": "medium", "category": "reasoning"},
    {"prompt": "Why do some programming languages use garbage collection while others require manual memory management?", "complexity": "medium", "category": "reasoning"},

    # medium / creative
    {"prompt": "Write a 3-sentence product description for a standing desk.", "complexity": "medium", "category": "creative"},
    {"prompt": "Draft a short intro paragraph for a blog post about productivity habits.", "complexity": "medium", "category": "creative"},

    # medium / code
    {"prompt": "Write a Python function that counts word frequency in a string.", "complexity": "medium", "category": "code"},
    {"prompt": "How do I connect to a PostgreSQL database from Node.js?", "complexity": "medium", "category": "code"},

    # complex / factual
    {"prompt": "Provide a detailed explanation of how the Federal Reserve sets monetary policy, including open market operations, reserve requirements, and forward guidance.", "complexity": "complex", "category": "factual"},
    {"prompt": "Explain in depth how the HTTPS protocol works, covering TLS handshake, certificate chains, symmetric/asymmetric encryption, and session keys.", "complexity": "complex", "category": "factual"},

    # complex / reasoning
    {"prompt": "Evaluate whether a two-sided marketplace should charge buyers, sellers, or both. Consider network effects, price sensitivity, competitive dynamics, and long-term platform health.", "complexity": "complex", "category": "reasoning"},
    {"prompt": "Analyze how rising interest rates affect equity valuations, corporate debt, consumer spending, and currency strength across an interconnected global economy.", "complexity": "complex", "category": "reasoning"},

    # complex / creative
    {"prompt": "Write the opening 500 words of a dystopian story where governments have outlawed personal privacy. Establish the world, tone, and a compelling protagonist.", "complexity": "complex", "category": "creative"},
    {"prompt": "Draft a comprehensive go-to-market memo for a B2B HR analytics platform entering the European market, covering positioning, channels, pricing, and a 6-month launch plan.", "complexity": "complex", "category": "creative"},

    # complex / code
    {"prompt": "Implement a generic event bus in TypeScript with typed events, subscriber management, and support for async handlers with error isolation.", "complexity": "complex", "category": "code"},
    {"prompt": "Design a multi-level caching strategy for a high-traffic read API: in-process cache, Redis, and database. Include invalidation logic, cache stampede prevention, and TTL reasoning.", "complexity": "complex", "category": "code"},
]
