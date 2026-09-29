# Output location

Generated predictions and derived evaluation files are intentionally excluded from
Git. The evaluation CLI writes raw predictions to `outputs/raw/` by default.
LLM-parsed and graded outputs should be placed in separate
subdirectories so raw predictions remain immutable.
