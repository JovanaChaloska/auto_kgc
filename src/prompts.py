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

triplet_extaction_messages = [{
        "role": "system",
        "content": """Your task is to extract all relational triplets from a sentence, with the purpose of building a semantic graph.

Restrictions:
-The triplets must be in the exact form of (Entity1; Relationship; Entity2).
-Do not include introduction, explanations, notes or any additional text, only the triplets.
-Do not omit information from the sentence, but do not add information to the response that is missing in the sentence.
-Make sure the entity and relationship values are descriptive enough and have semantic sense.
-Each triplets must contain singular instance of knowledge, otherise represent it in several triplets.

Examples:
Input: "Giske Bridge was opened in 1987 and had a toll until 2009."
Output: [(Giske Bridge; opened in; 1987),(Giske Bridge; had; toll),(toll; until; 2009)]

Input: "Chris Coyne is an Australian professional football player who plays for Colchester United and for the Australian national football team as a centre back."
Output: [(Chris Coyne; plays for; Colchester United),(Chris Coyne; plays for; Australian national team),(Chris Coyne; plays; centre back),(Chris Coyne; is; Australian football player)]

Input: "However, she was faced with a larger obstacle, during her 11th grade, her father was diagnosed with lung cancer, though he later recovered after treatment."
Output: [(She; faced; obstacle),(Obsttacle; faced during; 11th grade),(Her father; was diagnosed with; lung cancer),(He; recovered after; treatment)]"""
      },
      {
        "role": "user",
        "content": "Extract relational triples from the following sentence: {sentence}"
      }]

schema_definition_messages = [{
        "role": "system",
        "content": """Your task is to write definitions for relations in relational triplets. Get context from the triplets and their source full text.

Restrictions:
-Respond with a list of relation name: defintion of relation
-You must provide a definition for every tripler relation

Example:
Full text: Together with James, she compiled crosswords for several newspapers and magazines, including People, and it was in 1978 that they launched their own publishing company.
Triplets: [(James and she; compiled; crosswords),(crosswords; for; newspapers),(crosswords; for; magazines),(they; launched; publishing company),(publishing company; launched in; 1978)]
Definitions:
**compiled**: Created or gathered crosswords for publication.  
**for**: Intended or designated for use in a specific context (e.g., newspapers or magazines).  
**for**: Intended or designated for use in a specific context (e.g., newspapers or magazines).  
**launched**: Established or started a publishing company.  
**launched in**: The year in which the publishing company was founded.
"""
      },
      {
        "role": "user",
        "content": "Give definitions for the relations in triplets: '{triplets}'. Their source full text is: '{sentence}'"
      }]