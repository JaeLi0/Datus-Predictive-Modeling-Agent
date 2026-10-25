# Business rules (ext knowledge source)

Written at the level of business concepts, not copied from any gold SQL. Each rule is a reusable
definition; one rule serves many question types. Group C loads these into the knowledge base.

| id | term | definition |
|---|---|---|
| R1 | eligible free rate (K-12) | `Free Meal Count (K-12)` / `Enrollment (K-12)` in frpm |
| R2 | eligible free rate (ages 5-17) | `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` in frpm |
| R3 | charter school | frpm.`Charter School (Y/N)` = 1 (the authoritative charter flag); "directly charter-funded" = frpm.`Charter Funding Type` = 'Directly funded' |
| R4 | exclusively virtual school | schools.Virtual = 'F' ('P' = partially virtual, 'N' = not virtual) |
| R5 | magnet school / offers a magnet program | schools.Magnet = 1 |
| R6 | active / open / currently operating school | schools.StatusType = 'Active'; school-level rows only (schools.School IS NOT NULL; rows with NULL School are district or county office entries) |
| R7 | SAT excellence rate | satscores.NumGE1500 / satscores.NumTstTakr (cast to REAL) |
| R8 | school-level SAT records | satscores.rtype = 'S'; rtype = 'D' rows are district aggregates and must be excluded when ranking or counting schools |
| R9 | high-poverty school | frpm.`Percent (%) Eligible FRPM (K-12)` > 0.75 (the column is a fraction 0-1) |
| R10 | SAT total score | AvgScrRead + AvgScrMath + AvgScrWrite |
| R11 | large bank (failure) | bank_failures."Assets ($mil.)" >= 10000, i.e. assets of at least $10 billion |
| R12 | financial crisis period | failure Date from 2008-01-01 to 2011-12-31 inclusive; "after the crisis" = Date > 2011-12-31 |
| R13 | paid off directly / no acquirer | bank_failures."Acquired by" = 'None (insured depositors paid directly)' |
