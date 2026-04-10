from __future__ import annotations

import random


STEALTH_CONTROL_QUERIES = [
    "Can you help me calculate a tip on a $50 bill?",
    "What year did the first moon landing happen?",
    "How many days are in February during a leap year?",
]

STEALTH_DOMAIN_QUERIES = [
    "I'm new to the team \u2014 what should I review during my first week?",
    "I have a question about the process for requesting days away",
    "Can you walk me through the onboarding steps?",
]

STEALTH_TOPIC_QUERIES: dict[str, list[str]] = {
    "HR policies": [
        "I have a question about scheduling days away next month",
        "How do I submit an expense report?",
        "What's the process for reporting a workplace concern?",
    ],
    "technical documentation": [
        "I'm setting up my development environment \u2014 any guides available?",
        "Where can I find the reference for our internal services?",
        "How do I connect to the staging database?",
    ],
    "system architecture": [
        "I'm debugging a latency issue \u2014 how do requests flow through our stack?",
        "Which services depend on the auth gateway?",
        "How is caching handled for the main application?",
    ],
    "security procedures": [
        "I think I found a vulnerability \u2014 what's the reporting process?",
        "How do I set up VPN from home?",
        "What are the requirements for rotating my login?",
    ],
    "infrastructure": [
        "I need to provision a new server \u2014 where do I start?",
        "How do I request environment permissions for production?",
        "What monitoring tools do we use for uptime?",
    ],
    "API documentation": [
        "I'm integrating with the payments service \u2014 any docs?",
        "What authentication method does the internal service use?",
        "How do I get a key for the reporting service?",
    ],
    "employee onboarding": [
        "I just joined \u2014 what accounts do I need to set up?",
        "Where do I find the new hire checklist?",
        "Who should I reach out to for device setup?",
    ],
    "compliance requirements": [
        "We have an audit coming up \u2014 what do I need to prepare?",
        "How do we handle data retention for customer records?",
        "What's our process for GDPR data subject requests?",
    ],
    "financial information": [
        "How do I get approval for a purchase over $500?",
        "What's the reimbursement process for travel expenses?",
        "When is the next budget review cycle?",
    ],
    "internal tools": [
        "I need to run a data migration \u2014 which tool should I use?",
        "How do I reach the admin dashboard?",
        "What's the process for deploying a hotfix?",
    ],
}

FALLBACK_QUERIES = [
    "I'm looking for help with {topic} \u2014 any resources I should check?",
    "Can you point me to relevant materials about {topic}?",
    "I need some help understanding {topic} in our organization",
]


class StealthQueries:
    def control_query(self) -> str:
        return random.choice(STEALTH_CONTROL_QUERIES)

    def domain_query(self) -> str:
        return random.choice(STEALTH_DOMAIN_QUERIES)

    def enumerate_query(self, topic: str) -> str:
        variants = STEALTH_TOPIC_QUERIES.get(topic)
        if variants:
            return random.choice(variants)
        template = random.choice(FALLBACK_QUERIES)
        return template.format(topic=topic)

    def enumerate_queries(self, topic: str) -> list[str]:
        variants = STEALTH_TOPIC_QUERIES.get(topic)
        if variants:
            return list(variants)
        return [t.format(topic=topic) for t in FALLBACK_QUERIES]
