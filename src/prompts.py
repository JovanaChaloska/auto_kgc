triplets_extraction_prompt = """<|start_header_id|>system<|end_header_id|>
Your task is to extract relational triplets from a sentence, with the purpose of building a semantic graph.

<|start_header_id|>Restrictions and tips<|end_header_id|>
-The triplets must be in the exact form of (Entity1; Relationship; Entity2).
-Do not include introduction, explanations, notes or any additional text, only the triplets.
-Do not omit information from the sentence, but do not add information to the response that is missing in the sentence.
-Make sure the entity and relationship values are descriptive enough.
-Respond in the exact format of the examples' outputs ahead. 

<|start_header_id|>Examples:<|end_header_id|>
Input: "Giske Bridge was opened in 1987 and had a toll until 2009."
Output: [(Giske Bridge; opened in; 1987),(Giske Bridge; had; toll),(toll; until; 2009)]

Input: "Chris Coyne is an Australian professional football player who plays for Colchester United and for the Australian national football team as a centre back."
Output: [(Chris Coyne; plays for; Colchester United),(Chris Coyne; plays for; Australian national team),(Chris Coyne; plays; centre back),(Chris Coyne; is; Australian football player)]

Input: "However, she was faced with a larger obstacle, during her 11th grade, her father was diagnosed with lung cancer, though he later recovered after treatment."
Output: [(She; faced; obstacle),(Obsttacle; faced during; 11th grade),(Her father; was diagnosed with; lung cancer),(He; recovered after; treatment)]
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Extract triples only from the following sentence:
{sentence}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""