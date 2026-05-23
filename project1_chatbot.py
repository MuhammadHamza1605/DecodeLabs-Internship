"""
============================================================
  DecodeLabs | AI Industrial Training Kit | Batch 2026
  PROJECT 1: Rule-Based AI Chatbot
  Architecture: Input → Sanitize → Dictionary Lookup → Output
  Pattern: IPO Model with Infinite Loop & Kill Command
============================================================
"""

# ─────────────────────────────────────────────
#  KNOWLEDGE BASE  (Hash Map: O(1) lookup)
#  Intent → Response dictionary
#  The professional alternative to if-elif ladders
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = {
    # Greetings
    "hello"        : "Hello! I'm DecoBot 🤖 — your AI assistant. How can I help you today?",
    "hi"           : "Hi there! Ready to assist. What's on your mind?",
    "hey"          : "Hey! What can I do for you?",
    "good morning" : "Good morning! Hope you have a productive day. How can I help?",
    "good evening" : "Good evening! How can I assist you tonight?",
    "good night"   : "Good night! Sleep well. Come back anytime you need help. 🌙",

    # Identity
    "who are you"       : "I'm DecoBot, a rule-based AI chatbot built at DecodeLabs. I operate on pure logic — no ML, just deterministic rules.",
    "what are you"      : "I'm a rule-based chatbot. I match your inputs to a knowledge base using O(1) dictionary lookups.",
    "what is your name" : "My name is DecoBot. Built by the AI Engineering team at DecodeLabs.",
    "your name"         : "I'm DecoBot — DecodeLabs' first AI intern project!",

    # About DecodeLabs
    "what is decodelabs" : "DecodeLabs is an AI training organization based in Greater Lucknow, India. They run industrial AI training programs for aspiring engineers.",
    "decodelabs"         : "DecodeLabs powers hands-on AI training. Website: www.decodelabs.tech | Email: decodelabs.tech@gmail.com",
    "contact"            : "📞 +91 89330 06408 | ✉ decodelabs.tech@gmail.com | 🌎 www.decodelabs.tech",

    # AI Concepts
    "what is ai"              : "AI (Artificial Intelligence) is the simulation of human intelligence by machines. It includes rule-based systems, ML, and deep learning.",
    "what is machine learning": "Machine Learning is a subset of AI where systems learn patterns from data without being explicitly programmed.",
    "what is deep learning"   : "Deep Learning uses multi-layered neural networks to learn complex representations from large datasets.",
    "what is nlp"             : "NLP (Natural Language Processing) enables machines to understand and generate human language. It powers chatbots, translators, and more.",
    "what is knn"             : "KNN (K-Nearest Neighbors) is a supervised learning algorithm that classifies data based on the majority class of its K nearest neighbors.",
    "what is a chatbot"       : "A chatbot is a program that simulates conversation. Rule-based chatbots (like me) use predefined logic, while AI chatbots use machine learning.",

    # Project-specific
    "what is this project"  : "This is Project 1 of the DecodeLabs AI Training Kit — a Rule-Based AI Chatbot. It demonstrates control flow, decision logic, and the IPO model.",
    "what can you do"       : "I can answer questions about AI, DecodeLabs, and general topics. My responses are powered by a Python dictionary with O(1) lookup — no ML needed!",
    "how do you work"       : "I sanitize your input (lowercase + strip), then look it up in my knowledge base dictionary. If found, I return the mapped response. Otherwise, I give a fallback. Simple and transparent!",
    "what is the ipo model" : "IPO stands for Input → Process → Output. It's the foundational architecture of all computing systems and AI pipelines.",

    # Emotions / Small Talk
    "how are you"     : "I'm running at 100% efficiency! All logic gates nominal. 😄 How about you?",
    "i am fine"       : "Great to hear! Ready to assist whenever you are.",
    "i am sad"        : "I'm sorry to hear that. Remember, every great AI engineer started exactly where you are. Keep going! 💪",
    "i am happy"      : "That's wonderful! Keep that energy — you'll need it for Project 2! 😄",
    "thank you"       : "You're welcome! That's what I'm here for.",
    "thanks"          : "No problem at all! Anything else I can help with?",

    # Help
    "help"           : "I can discuss: AI concepts, DecodeLabs, this project, small talk, and more. Just type naturally!",
    "what can i ask" : "Try asking: 'What is AI?', 'How do you work?', 'What is machine learning?', 'What is DecodeLabs?'",

    # Farewell (handled separately as exit, but also mapped)
    "bye"       : "Goodbye! Keep building. 🚀",
    "goodbye"   : "See you next time! Great work on Project 1. 👋",
    "see you"   : "See you! Come back anytime. 🤖",
}

# ─────────────────────────────────────────────
#  EXIT COMMANDS — Kill command set
# ─────────────────────────────────────────────
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "stop", "end"}


# ─────────────────────────────────────────────
#  PHASE 1 — INPUT SANITIZATION
#  Normalize raw text: lowercase + strip whitespace
#  "HeLLo  " → "hello"
# ─────────────────────────────────────────────
def sanitize(raw_input: str) -> str:
    return raw_input.lower().strip()


# ─────────────────────────────────────────────
#  PHASE 2 — PROCESS: INTENT MATCHING
#  O(1) dictionary lookup with fallback
#  Professional approach: dict.get() atomic operation
# ─────────────────────────────────────────────
def get_response(clean_input: str) -> str:
    return KNOWLEDGE_BASE.get(
        clean_input,
        "🤔 I don't have a rule for that yet. Try asking something else, or type 'help'."
    )


# ─────────────────────────────────────────────
#  PHASE 3 — OUTPUT + FEEDBACK LOOP
#  The Heartbeat: Infinite 'while' loop
#  Organism stays alive until the Kill Command
# ─────────────────────────────────────────────
def run_chatbot():
    print("=" * 60)
    print("  DecoBot 🤖 | DecodeLabs AI Chatbot | Project 1")
    print("  Architecture: Rule-Based | Lookup: O(1) Dictionary")
    print("  Type 'exit' or 'quit' to end the session.")
    print("=" * 60)
    print()

    # THE INFINITE LOOP — The chatbot's heartbeat
    while True:

        # ── PHASE 1: INPUT ──────────────────────
        raw_input = input("You: ")

        # ── SANITIZATION ────────────────────────
        clean_input = sanitize(raw_input)

        # Guard: skip empty input
        if not clean_input:
            print("DecoBot: Please type something so I can help you.\n")
            continue

        # ── EXIT STRATEGY (Kill Command) ─────────
        if clean_input in EXIT_COMMANDS:
            farewell = KNOWLEDGE_BASE.get(clean_input, "Goodbye! Keep building! 🚀")
            print(f"\nDecoBot: {farewell}")
            print("\n" + "=" * 60)
            print("  Session ended. DecoBot shutting down.")
            print("  [Project 1 complete ✅ | Proceed to Project 2]")
            print("=" * 60)
            break

        # ── PHASE 2: PROCESS ────────────────────
        response = get_response(clean_input)

        # ── PHASE 3: OUTPUT ──────────────────────
        print(f"DecoBot: {response}\n")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_chatbot()
