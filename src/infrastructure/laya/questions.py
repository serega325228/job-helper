MATCH_QUESTIONS = {
    "role_fit": {
        "type": "choice",
        "instructions": (
            "Evaluate how closely the vacancy role corresponds "
            "to the candidate's target roles."
        ),
        "criteria": {
            "none": "The roles are unrelated",
            "weak": "Some overlap, but substantially different work",
            "good": "Same general role with some differences",
            "strong": "Direct match with the candidate's target role",
        },
    },

    "skill_fit": {
        "type": "choice",
        "instructions": (
            "Evaluate the candidate's technical compatibility "
            "with the vacancy requirements."
        ),
        "criteria": {
            "none": "Almost no important requirements are satisfied",
            "weak": "Some relevant skills but major gaps exist",
            "good": "Most important requirements are satisfied",
            "strong": "Almost all important requirements are satisfied",
        },
    },
}
