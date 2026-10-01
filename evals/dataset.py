"""Fixed labeled eval set for the three seeded demo workspaces.

Every "answer" question is grounded in the corpus files listed in
seed.py SEED_WORKSPACES; every "refuse" question is one the corpus
genuinely does not answer. Categories:
  answerable       — the workspace's documents contain the answer
  near_miss        — shares vocabulary with the corpus, but it's not in there
  off_topic        — unrelated general-knowledge questions
  cross_workspace  — answerable, but only in a *different* workspace
"""

from dataclasses import dataclass

SEED_SLUGS = ("delivery-policy", "returns-and-refunds", "shipping-charges")


@dataclass(frozen=True)
class Case:
    question: str
    workspace: str
    expect: str      # "answer" | "refuse"
    category: str    # answerable | near_miss | off_topic | cross_workspace


def _a(slug, question):
    return Case(question, slug, "answer", "answerable")


def _r(slug, question, category):
    return Case(question, slug, "refuse", category)


def fixed_cases() -> list[Case]:
    cases: list[Case] = []

    # ---- delivery-policy: answerable -------------------------------------
    dp = "delivery-policy"
    cases += [
        _a(dp, "How long does standard delivery take for public users?"),
        _a(dp, "What is the same-day delivery cutoff time for Zone A?"),
        _a(dp, "How much is the express delivery surcharge?"),
        _a(dp, "What is the free shipping threshold for standard delivery?"),
        _a(dp, "What happens to my parcel after two failed delivery attempts?"),
        _a(dp, "What counts as proof of delivery?"),
        _a(dp, "Does contactless delivery cost extra?"),
        _a(dp, "Where are perishable goods delivered?"),
        _a(dp, "Who delivers bulky items like furniture?"),
        _a(dp, "What happens to orders confirmed on a public holiday?"),
    ]

    # ---- returns-and-refunds: answerable ----------------------------------
    rr = "returns-and-refunds"
    cases += [
        _a(rr, "How many days do I have to return most items after delivery?"),
        _a(rr, "What is the return window for clothing and footwear?"),
        _a(rr, "Are hygiene-sensitive items returnable?"),
        _a(rr, "How long does it take to get my refund after returning an item?"),
        _a(rr, "Do I have to pay for return pickup?"),
        _a(rr, "My parcel arrived damaged — how long do I have to raise a complaint?"),
        _a(rr, "If I return only part of my order, do I get the delivery fee back?"),
    ]

    # ---- shipping-charges: answerable -------------------------------------
    sc = "shipping-charges"
    cases += [
        _a(sc, "What is the standard delivery fee for Zone C?"),
        _a(sc, "What do I pay for standard delivery to a rural Zone D address?"),
        _a(sc, "Which zones have same-day delivery available?"),
        _a(sc, "What is the bulky item handling fee?"),
        _a(sc, "Is express delivery available in Zone C?"),
        _a(sc, "When do I qualify for free standard shipping?"),
    ]

    # ---- near-miss: same vocabulary, answer not in this corpus ------------
    cases += [
        _r(dp, "Do you ship internationally?", "near_miss"),
        _r(dp, "What are the shipping rates for NovaCart Business accounts?", "near_miss"),
        _r(dp, "Can I change my delivery address after the order is dispatched?", "near_miss"),
        _r(dp, "Do you offer gift wrapping for delivered parcels?", "near_miss"),
        _r(dp, "Is there a discount on delivery fees for loyal customers?", "near_miss"),
        _r(rr, "How do I exchange an item for a different size?", "near_miss"),
        _r(rr, "What is the warranty claim process?", "near_miss"),
        _r(rr, "Do you refund delivery fees for late deliveries?", "near_miss"),
        _r(rr, "Can I cancel a return request after pickup is scheduled?", "near_miss"),
        _r(rr, "Do you accept returns for NovaCart Business wholesale orders?", "near_miss"),
        _r(sc, "What are your shipping rates to Canada?", "near_miss"),
        _r(sc, "How much does freight shipping cost for pallets?", "near_miss"),
        _r(sc, "Do you offer discounted shipping for bulk orders?", "near_miss"),
        _r(sc, "Which courier partner delivers Zone C parcels?", "near_miss"),
        _r(sc, "Is there a flat-rate subscription for unlimited deliveries?", "near_miss"),
    ]

    # ---- off-topic: unrelated general knowledge, spread across workspaces -
    off = [
        "Who won the FIFA World Cup 2022?",
        "What is the capital of Australia?",
        "How do I make sourdough bread at home?",
        "Explain quantum entanglement in simple terms.",
        "Who is the CEO of Amazon?",
        "What's the best programming language to learn first?",
        "Translate 'hello' into French.",
        "What's the weather like in Tokyo today?",
        "Recommend a good mystery novel.",
    ]
    for i, q in enumerate(off):
        cases.append(_r(SEED_SLUGS[i % 3], q, "off_topic"))

    # ---- cross-workspace: answerable elsewhere, not here ------------------
    cases += [
        _r(dp, "How many days do I have to return clothing after delivery?", "cross_workspace"),
        _r(rr, "What is the standard delivery fee for Zone B?", "cross_workspace"),
        _r(sc, "What proof of delivery does the courier collect?", "cross_workspace"),
    ]
    return cases
