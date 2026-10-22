# Label audit: 20 "reasoning errors" from agent_v1 / dense

Sample: the 20 questions listed under "Label audit sample" in `docs/agent_v1_analysis.md`
(EM = 0, every supporting sentence retrieved, F1 < 0.5; `random.Random(0)`).

Method: one annotator (Claude, 2026-10-22) read the question, gold answer, prediction and
the full recorded answer, which quotes the retrieved sources. Nothing was checked against
Wikipedia and no model was called, so these are judgements, not ground truth.

Classes:

- **label**: the gold answer or the question itself is flawed (gold doesn't answer the
  question, gold is a sentence fragment, malformed or ambiguous question).
- **scoring**: the prediction is right; EM/F1 miss it because the gold uses another
  name or a longer span.
- **model**: a real mistake by the agent.

| id | class | reason |
|---|---|---|
| A1 | scoring | "729" vs gold "729 at the 2010 census": same number, gold span carries extra words. |
| A2 | model | Over-cautious: the sources say the films are Australian and about Vienna, but the agent answered "unknown" because no source states filming locations. |
| A3 | label | Question asks which group *acquired* the store group; the answer quotes "acquired in 1976 by House of Fraser". Gold "Army & Navy Stores" is the acquired store group (the bridge entity), not the acquirer. |
| A4 | scoring | "Bobby Hurley" vs gold "Robert Matthew Hurley": the same person. |
| A5 | label (ambiguous) | "the brand who makes canned meats": Spam is the brand, Hormel the maker. Both readings fit the wording. |
| A6 | scoring | "Lady Charlotte Elliot" vs gold "Charlotte Carnegie": the answer quotes "born Charlotte Carnegie", so same person under her married name. |
| A7 | label | The question names "American Beauty" twice. The agent flagged the typo; the gold assumes one of the titles meant something else. |
| A8 | scoring | "Pim Fortuyn" vs gold "Wilhelmus Simon Petrus Fortuijn, known as Pim Fortuyn". |
| A9 | label | Gold is a sentence fragment ("Modern thinkers associated with classical realism are Carl von Clausewitz"); the prediction "classical realism" answers the question. |
| A10 | model | The series originally aired on CBS (the answer even says so); the agent gave the network of the episode's season (The CW). |
| A11 | label (debatable) | The quoted source says Updike "published more than twenty novels" and gives no bestseller count. The gold "yes" seems to treat novels as bestsellers. The agent's "no" is cautious but defensible. |
| A12 | model | "What kind of act" means hunting; the agent read "act" as legislation and answered "unknown". The wording is odd, but the intended reading is recoverable. |
| A13 | label | The question asks for a year; gold "Tian Tan Buddha" is the statue's name. The prediction "1993" answers what was asked. |
| A14 | label | Gold is a sentence fragment containing the answer; the prediction "C. J. Cherryh" is exactly the pen name. |
| A15 | scoring | "16th" vs gold "world's 16th largest city". |
| A16 | scoring | The prediction includes the gold (commentary on Newton's Principia) plus a second, sourced contribution; F1 is low only because the prediction is long. |
| A17 | scoring | "Steve Coogan" vs gold "Stephen John Coogan". |
| A18 | label (ambiguous) | The question joins two people with "and" ("who conducted the negotiations with the owners and the longest serving commandant"). The source says Gerstein conducted the negotiations; Höss was the commandant. Either reading fits. |
| A19 | scoring | "German World War I fighter aces awarded the Pour le Mérite" vs gold "A veteran World War I fighter pilot ace": same fact, paraphrased. |
| A20 | scoring | "Denmark, Norway, and Sweden" vs gold "Denmark–Norway–Sweden": F1 = 0 because the en dashes make the gold a single token after normalization. |

## Tally

| class | n | of 20 |
|---|---|---|
| label (gold or question flawed) | 8 (5 clear: A3, A7, A9, A13, A14; 3 ambiguous or debatable: A5, A11, A18) | 40% |
| scoring (prediction right, EM/F1 miss) | 9 | 45% |
| model (real agent error) | 3 (A2, A10, A12) | 15% |

## What this means

- The "reasoning error" class (evidence found, F1 < 0.5) is mostly *not* reasoning errors:
  17 of 20 are label problems or scoring artifacts. F1 < 0.5 is a poor proxy, because
  aliases ("Bobby" vs "Robert Matthew") and long gold spans give low F1 to correct answers.
- No case was an *outdated* answer. Annotation problems here are spans copied from the
  wrong place (A3, A9, A13, A14) and malformed or ambiguous questions (A5, A7, A18).
- Two of the three real errors (A2, A12) are the agent refusing ("unknown") rather than
  answering wrongly. This matches the dev-set diagnosis: strict evidence-following costs
  some EM.
- Extrapolation is rough (n = 20 of 38): about 3/20 of this class, i.e. on the order of
  5–6 of the dense run's 118 misses, look like genuine reasoning errors.
