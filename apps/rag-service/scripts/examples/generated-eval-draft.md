## gen-qa-0001

### Verdict

VALID

### Issues

- 原 question 过于抽象（relationship），容易导致回答发散
- 原 reference 略微压缩，未明确 “responsibilities” 这一关键中介

### Fix

- Question:
  According to the chunk, why may younger musicians find it easier to focus on their careers?

- Reference:
  The chunk explains that as people get older, they accumulate more responsibilities, while younger musicians have fewer obligations. This gives them more freedom to take risks and focus on their careers.

### Notes

- chunk 已验证与 source 完全一致（strong grounding）
- 属于高质量 conceptual QA
- 优化后更利于 RAGAS 稳定评分

## gen-qa-0002

### Verdict

VALID

### Issues

- Original question is slightly broad because "one challenge" leaves answer boundaries loose.
- Original reference is correct, but the key evidence is better expressed directly and more extractively.

### Fix

- Question:
  According to the chunk, what do young artists need to do to stand out in the niche country music market?

- Reference:
  Young artists need to create anticipation and make a whole lot of noise to stand out in the niche market.

### Grounding Truth

Young artists need to create anticipation and make a whole lot of noise to stand out.

### Notes

- Source grounding confirmed from the article text. :contentReference[oaicite:0]{index=0}
- This sample is best treated as factual / extractive QA. :contentReference[oaicite:1]{index=1}
- Adding `ground_truth_context` is recommended for future auditability and retrieval debugging.

## gen-qa-0003

### Verdict

VALID

### Issues

- Original question uses singular ("Which album"), but answer contains two → mismatch
- Minor wording noise in reference ("According to the source")

### Fix

- Question:
  Which albums are nominated for both Album of the Year and Best Pop Vocal Album?

- Reference:
  Taylor Swift’s Midnights and Miley Cyrus’ Endless Summer Vacation are both nominated for Album of the Year and Best Pop Vocal Album.

### Grounding Truth

Midnights is also in contention for Best Pop Vocal Album. Miley Cyrus’ Endless Summer Vacation is also nominated for Album of the Year... Endless Summer Vacation is also nominated for Best Pop Vocal Album.

### Notes

- Grounding verified from source text :contentReference[oaicite:0]{index=0}
- Clear factual multi-entity QA
- After fixing singular/plural mismatch, scoring becomes stable

## gen-qa-0004

### Verdict

VALID

### Issues

- Original question ("philosophy") slightly broad and abstract
- Original reference contains minor wording noise ("According to the source")

### Fix

- Question:
  According to the chunk, what is Zander Hulme's view on environmental design in games?

- Reference:
  Zander Hulme believes environmental design is not just about adding background ambience, but about creating a living world that helps sell the game and allows players to suspend their disbelief.

### Grounding Truth

"It's not just whacking in some background ambience. It's about making a living world that kind of helps you sell your game and helps the player to suspend their belief when they enter into your little world."

### Notes

- Grounding fully supported by source text :contentReference[oaicite:0]{index=0}
- Strong conceptual QA with clear extractive support
- Optimized for stable RAGAS scoring

## gen-qa-0005

### Verdict

VALID

### Issues

- Original question wording ("composed music for") is less precise than source wording ("received commissions from")
- Minor wording noise in reference ("According to the source")

### Fix

- Question:
  Which individual has received commissions from the Melbourne Symphony Orchestra?

- Reference:
  Amy Bastow has received commissions from the Melbourne Symphony Orchestra.

### Grounding Truth

Amy has received commissions from the Melbourne Symphony Orchestra.

### Notes

- Grounding directly supported by source text :contentReference[oaicite:0]{index=0}
- Clear entity lookup QA with exact answer span
- Optimized wording improves precision and reduces ambiguity

## gen-qa-0006

### Verdict

VALID

### Issues

- Original reference slightly verbose and over-explained
- Contains redundant phrasing ("According to the source")
- Includes more detail than needed for evaluation stability

### Fix

- Question:
  According to APRA AMCOS and NATSIMO, what is the main criticism of the Productivity Commission's interim report?

- Reference:
  They criticise the report for laying the groundwork to legitimise widespread theft of Australian works by AI companies and for perpetuating the myth that copyright is a barrier to innovation.

### Grounding Truth

They're laying the groundwork to legitimise what they themselves acknowledge is already widespread theft... The Commission's recommendation also perpetuates the thoroughly debunked myth that copyright is a barrier to innovation.

### Notes

- Grounding clearly supported by source statements :contentReference[oaicite:0]{index=0}
- Multi-point conceptual QA (2 key claims)
- Well-suited for testing reasoning + aggregation
- Optimized for stable RAGAS scoring

## gen-qa-0007

### Verdict

NEED FIX

### Issues

- Question asks for "time and date", but reference only contains time → mismatch
- Missing date in answer leads to evaluation inconsistency

### Fix

- Question:
  What is the time of the Anatomy of an Album event with Ruel & M-Phazes?

- Reference:
  The event takes place from 4 - 6pm AEST.

### Grounding Truth

Time: 4 - 6pm AEST

### Notes

- Grounding directly supported by source text
- This is a factual event-detail QA
- Important to ensure question-answer scope alignment for stable evaluation

## gen-qa-0008

### Verdict

VALID

### Issues

- Original reference includes unnecessary phrasing ("According to the source")

### Fix

- Question:
  What is the standard notice period required to resign from APRA membership?

- Reference:
  The standard notice period is six months.

### Grounding Truth

There is a standard six-month notice period for resignation of membership.

### Notes

- Grounding directly supported by source text :contentReference[oaicite:0]{index=0}
- Clean factual extractive QA
- Highly stable for RAGAS evaluation

## gen-qa-0009

### Verdict

VALID

### Issues

- Original reference includes unnecessary phrasing ("According to the source")

### Fix

- Question:
  How many workers from the Philippines, Vietnam, and Pakistan provided services to APRA during the reporting period?

- Reference:
  23 workers provided services to APRA.

### Grounding Truth

23 workers located in these countries provided services to APRA.

### Notes

- Grounding directly supported by source text :contentReference[oaicite:0]{index=0}
- Clean factual extractive QA (numeric answer)
- Highly stable for evaluation (low ambiguity)

## gen-qa-0010

### Verdict

VALID (after fix)

### Issues

- Original question is too open-ended ("best"), leading to unstable answer boundaries
- Answer could vary (moment vs example vs full description)

### Fix

- Question:
  According to the chunk, what is described as the best SXSW memory?

- Reference:
  The best SXSW memory is the moment when you realise you have just witnessed something special, such as seeing bands for the first time in small rooms with like-minded fans, including seeing Chet Faker perform in a conference room.

### Grounding Truth

Best SXSW memory: That moment when you realise you just witnessed something special... like seeing Chet Faker in the conference room doing three songs on a keyboard.

### Notes

- Grounding clearly supported by source text
- Conceptual + descriptive QA (experience-based)
- Slightly less stable than factual QA due to subjective phrasing
- Acceptable after constraining answer scope

## gen-qa-0011

### Verdict

VALID

### Issues

- Original reference includes unnecessary phrasing ("According to the source")
- Includes vague wording ("and more!") which is not needed for evaluation

### Fix

- Question:
  What awards did Julian Wilton and Narayana Johnson win at the Australian Game Developer Awards (AGDA)?

- Reference:
  They won Best Game and Excellence In Music.

### Grounding Truth

they proceeded to win Best Game and Excellence In Music (and more!) at the Australian Game Developer Awards (AGDA).

### Notes

- Grounding directly supported by source text :contentReference[oaicite:0]{index=0}
- Multi-entity factual QA (list answer)
- Removing "and more" improves answer precision and scoring stability

## gen-qa-0012

### Verdict

VALID

### Issues

- Original reference includes unnecessary phrasing ("According to the source")
- Question slightly verbose ("as stated in the text") — not needed

### Fix

- Question:
  What are APRA’s core values?

- Reference:
  APRA’s core values are collaboration, respect, skill, imagination, and accountability.

### Grounding Truth

the APRA’s core values of collaboration, respect, skill, imagination and accountability.

### Notes

- Grounding directly supported by source text :contentReference[oaicite:0]{index=0}
- List-type factual QA (enumeration)
- Stable and unambiguous answer

## gen-qa-0013

### Verdict

VALID

### Issues

- Original reference includes unnecessary phrasing ("According to the source")
- Answer can be simplified to exact entity for stronger evaluation precision

### Fix

- Question:
  Who did the speaker employ as a co-composer on a documentary?

- Reference:
  Belinda.

### Grounding Truth

I have employed Belinda as a co-composer on a documentary.

### Notes

- Grounding directly supported by source text :contentReference[oaicite:0]{index=0}
- Clean entity lookup QA (single-token answer)
- Highly stable and ideal for exact-match evaluation

## gen-qa-0014

### Verdict

VALID

### Issues

- Original reference includes unnecessary phrasing ("According to the source")
- Answer can be simplified to exact entities for better evaluation precision

### Fix

- Question:
  Who were the two most influential mentors mentioned in the text?

- Reference:
  Martin Armiger and Michael A. Levine.

### Grounding Truth

2 that have been the most influential are Martin Armiger ... and Michael A. Levine.

### Notes

- Grounding directly supported by source text :contentReference[oaicite:0]{index=0}
- Multi-entity entity lookup QA
- Stable answer with clearly defined entities

## gen-qa-0015

### Verdict

VALID

### Issues

- Original reference includes unnecessary phrasing ("According to the source")

### Fix

- Question:
  What is the purpose of the $8,000 honorarium provided by NATSIMO?

- Reference:
  It is provided to cover living expenses and travel costs incurred from participating in the program.

### Grounding Truth

to cover living expenses and travel costs incurred as a result of participating in the program.

### Notes

- Grounding directly supported by source text
- Clear factual purpose-based QA
- Stable and unambiguous answer

## gen-qa-0016

### Verdict

VALID (high-value)

### Issues

- Original reference slightly verbose
- Includes unnecessary phrasing ("According to the source")

### Fix

- Question:
  What change did Robbie Miller make to his music after observing his teenage niece's listening behaviour?

- Reference:
  He removed chord introductions and went straight to singing.

### Grounding Truth

I gave up four bars of chord introductions and went straight to singing.

### Notes

- Grounding directly supported by source text
- Requires light paraphrasing (not purely extractive)
- Tests causal understanding (observation → change)
- Higher value than simple factual QA

## gen-qa-0017

### Verdict

VALID (after fix)

### Issues

- Relationship direction is ambiguous ("paired with")
- Context is a flat list → requires correct parsing
- Potential confusion over mentor vs mentee roles

### Fix

- Question:
  Who is paired with Georgia Fields in the mentorship list?

- Reference:
  Sasha Gavlek.

### Grounding Truth

Sasha Gavlek with Georgia Fields

### Notes

- Grounding directly supported by source text
- Requires parsing structured list data
- Tests relationship extraction rather than simple retrieval
- Medium difficulty (higher than basic factual QA)

## gen-qa-0018

### Verdict

VALID (after fix)

### Issues

- Original question is open-ended ("one of"), allowing multiple valid answers
- Reference only captures one possible answer → evaluation instability
- Context includes multiple potential challenges

### Fix

- Question:
  According to Dean Ormston, what is a key challenge for the government in supporting Australia’s contemporary music industry?

- Reference:
  Developing a smart whole-of-government approach across sectors such as cultural diplomacy, trade, tourism, small business, education, health and arts.

### Grounding Truth

The challenge for government is to develop a smart whole-of-government approach across cultural diplomacy, trade, tourism, small business, education, health and arts.

### Notes

- Grounding directly supported by source text :contentReference[oaicite:0]{index=0}
- Original version suffers from answer ambiguity
- Fixed version constrains answer space → improves evaluation stability

## gen-qa-0019

### Verdict

VALID (after fix)

### Issues

- Original answer is overly long and includes too many granular details
- High risk of partial answers being incorrectly judged
- List is unstructured (mix of methods and platforms)

### Fix

- Question:
  What are the main ways recordings of TAFE events and activities can be shared?

- Reference:
  They can be shared via password-protected TAFE platforms, email or messaging systems, TAFE websites or official social media channels, and conferencing platforms such as Zoom.

### Grounding Truth

shared... via online, password protected TAFE platforms... email, private messaging system... TAFE website or official social media channels... conferencing platforms such as Zoom

### Notes

- Grounding supported by source text
- Requires grouping multiple items into categories
- Medium-to-high difficulty due to list abstraction
- Optimized to reduce evaluation instability

## gen-qa-0020

### Verdict

VALID (minor refinement)

### Issues

- Original reference includes unnecessary phrasing ("According to the source")
- Question includes year ("2026") which is implied but not strictly necessary

### Fix

- Question:
  What is the maximum funding amount available for artists through the Art Residencies program?

- Reference:
  $5,000.

### Grounding Truth

include up to $5,000 funding for artists.

### Notes

- Grounding directly supported by source text :contentReference[oaicite:1]{index=1}
- Numeric factual QA (high precision)
- Slight temporal ambiguity removed for stability

## gen-qa-0021

### Verdict

VALID

### Issues

- Original reference includes unnecessary phring ("According to the source")
- Question slightly verbose

### Fix

- Question:
  Which organisations are mentioned as supporting Australian and New Zealand artists?

- Reference:
  APRA AMCOS and triple j.

### Grounding Truth

support of organisations like APRA AMCOS and triple j

### Notes

- Grounding directly supported by source text :contentReference[oaicite:0]{index=0}
- Simple multi-entity factual QA
- Low ambiguity and highly stable

## gen-qa-0022

### Verdict

VALID

### Issues

- Original reference includes unnecessary phrasing ("According to the source")
- Answer should be simplified to a boolean for evaluation stability

### Fix

- Question:
  Is Hindley Street Music Hall wheelchair accessible?

- Reference:
  Yes.

### Grounding Truth

Hindley Street Music Hall is a wheelchair-accessible venue.

### Notes

- Grounding directly supported by source text
- Boolean QA (yes/no)
- Extremely stable and easy to evaluate

## gen-qa-0023

### Verdict

VALID

### Issues

- Reference includes unnecessary phrasing ("According to the source")
- Answer should be normalized into a clean list-style response

### Fix

- Question:
  What are the two types of music use reporting used for jingle distributions?

- Reference:
  Music recognition technology and the Jingle self-reporting form.

### Grounding Truth

We use two types of music use reporting for jingle distributions - Music recognition technology ... Jingle self-reporting form ...

### Notes

- Multi-entity extraction (list of 2 items)
- Explicitly stated in source → high confidence grounding
- Good test for:
  - list extraction
  - conjunction understanding ("and")

## gen-qa-0024

### Verdict

VALID (slightly higher-level factual)

### Issues

- Reference contains unnecessary phrasing ("According to the source")
- Minor verbosity ("many amazing creative people" → can normalize)

### Fix

- Question:
  What are the main benefits of being involved with APRA AMCOS according to the text?

- Reference:
  Royalty collection and the opportunity to meet creative people through events held throughout the year and online.

### Grounding Truth

There are so many benefits - the main ones being royalty collection and getting to meet so many amazing creative people ... through events held throughout the year and online.

### Notes

- Multi-benefit extraction (list)
- Slight paraphrasing required (not pure copy)
- Good test for:
  - summarisation from informal tone
  - extracting key points from descriptive text

## gen-qa-0025

### Verdict

VALID (high-quality factual extraction from complex policy text)

### Issues

- Reference contains unnecessary phrase ("According to the source")

### Fix

- Question:
  What is the eligibility requirement for the composer of a work submitted for the Performance of the Year category?

- Reference:
  The work must be composed by an Australian citizen or a composer who has permanent residence in Australia.

### Grounding Truth

The performed work must have been composed by an Australian citizen or by a composer who has permanent residence in Australia.

### Notes

- Requires filtering from a long policy document
- Tests ability to:
  - locate specific condition within structured rules
  - ignore irrelevant surrounding clauses
- Strong real-world applicability (eligibility criteria extraction)
