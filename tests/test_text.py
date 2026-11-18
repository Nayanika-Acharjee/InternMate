from internmate.text import extract_skills, job_skills, normalize_skill, strip_html, tokenize


def test_aliases_normalise():
    assert normalize_skill("Node.js") == "node"
    assert normalize_skill(" ReactJS ") == "react"
    assert normalize_skill("Postgres") == "postgresql"


def test_extract_skills_uses_word_boundaries():
    skills = extract_skills("Experience with Python, Node.js and Power BI. Excellent communication.")
    assert {"python", "node", "power bi"} <= skills
    assert "excel" not in skills  # "Excellent" must not match


def test_ambiguous_skills_only_from_tags():
    assert "go" not in job_skills("Intern", "You will go live with features", [])
    assert "go" in job_skills("Intern", "", ["Go"])


def test_tokenize_and_strip_html():
    assert tokenize("The C++ and Node.js role.") == ["c++", "node.js"]
    assert strip_html("<p>Hello&nbsp;<b>world</b></p>") == "Hello world"
