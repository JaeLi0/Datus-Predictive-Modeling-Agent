"""Build the EX evaluation set (base.jsonl) and the held-out paraphrase set (paraphrase.jsonl).

Each item: id, datasource, tier (single / join / rule), question, gold_sql, rules (business-rule ids
the question depends on, see eval/kb/rules.md), source (bird / new).

Gold SQL notes:
- satscores.cds lost its leading zero for county codes 01-09 (211 of 2269 rows have 13 chars), so a
  direct ``cds = CDSCode`` join silently drops those rows. Gold SQL joins on the normalized key
  ``printf('%014d', CAST(cds AS INTEGER))``. The 10 BIRD questions were rewritten accordingly.
- Questions are asked WITHOUT evidence; business definitions live in the knowledge base (group C).

Run: .venv/bin/python eval/datasets/build_dataset.py  (prints answer shapes for review)
"""

import json
import sqlite3
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
SQLITE = ROOT / "sample_data/california_schools/california_schools.sqlite"
DUCK = ROOT / "eval/kb/duckdb-demo-eval.duckdb"
OUT = Path(__file__).resolve().parent

SAT_JOIN = "printf('%014d', CAST(T1.cds AS INTEGER)) = T2.CDSCode"

CA = "california_schools"
BF = "demo"

BASE = [
    # ---------------- BIRD (10, rewritten join key) ----------------
    ("b00", CA, "rule", "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
     "SELECT MAX(`Free Meal Count (K-12)` / `Enrollment (K-12)`) FROM frpm WHERE `County Name` = 'Alameda'",
     ["R1"], "bird"),
    ("b01", CA, "rule", "Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.",
     "SELECT `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` AS r FROM frpm WHERE `Educational Option Type` = 'Continuation School' AND `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` IS NOT NULL ORDER BY r ASC LIMIT 3",
     ["R2"], "bird"),
    ("b02", CA, "rule", "Please list the zip code of all the charter schools in Fresno County Office of Education.",
     "SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`District Name` = 'Fresno County Office of Education' AND T1.`Charter School (Y/N)` = 1",
     ["R3"], "bird"),
    ("b03", CA, "join", "What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?",
     "SELECT T2.MailStreet FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T1.`FRPM Count (K-12)` DESC LIMIT 1",
     [], "bird"),
    ("b04", CA, "rule", "Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.",
     "SELECT T2.Phone FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`Charter Funding Type` = 'Directly funded' AND T1.`Charter School (Y/N)` = 1 AND T2.OpenDate > '2000-01-01'",
     ["R3"], "bird"),
    ("b05", CA, "rule", "How many schools with an average score in Math greater than 400 in the SAT test are exclusively virtual?",
     f"SELECT COUNT(DISTINCT T2.School) FROM satscores AS T1 INNER JOIN schools AS T2 ON {SAT_JOIN} WHERE T2.Virtual = 'F' AND T1.AvgScrMath > 400",
     ["R4"], "bird"),
    ("b06", CA, "rule", "Among the schools with the SAT test takers of over 500, please list the schools that are magnet schools or offer a magnet program.",
     f"SELECT T2.School FROM satscores AS T1 INNER JOIN schools AS T2 ON {SAT_JOIN} WHERE T2.Magnet = 1 AND T1.NumTstTakr > 500 AND T1.rtype = 'S'",
     ["R5", "R8"], "bird"),
    ("b07", CA, "join", "What is the phone number of the school that has the highest number of test takers with an SAT score of over 1500?",
     f"SELECT T2.Phone FROM satscores AS T1 INNER JOIN schools AS T2 ON {SAT_JOIN} WHERE T1.rtype = 'S' ORDER BY T1.NumGE1500 DESC LIMIT 1",
     [], "bird"),
    ("b08", CA, "join", "What is the number of SAT test takers of the school with the highest FRPM count for K-12 students?",
     f"SELECT T1.NumTstTakr FROM satscores AS T1 INNER JOIN frpm AS T2 ON {SAT_JOIN} ORDER BY T2.`FRPM Count (K-12)` DESC LIMIT 1",
     [], "bird"),
    ("b09", CA, "join", "Among the schools with the average score in Math over 560 in the SAT test, how many schools are directly charter-funded?",
     f"SELECT COUNT(*) FROM satscores AS T1 INNER JOIN frpm AS T2 ON {SAT_JOIN} WHERE T1.AvgScrMath > 560 AND T2.`Charter Funding Type` = 'Directly funded'",
     [], "bird"),
    # ---------------- california_schools: single table ----------------
    ("c01", CA, "single", "Which county has the most schools listed in the FRPM data?",
     "SELECT `County Name` FROM frpm GROUP BY `County Name` ORDER BY COUNT(*) DESC LIMIT 1", [], "new"),
    ("c02", CA, "single", "What is the total K-12 enrollment of all schools in San Diego County according to the FRPM data?",
     "SELECT SUM(`Enrollment (K-12)`) FROM frpm WHERE `County Name` = 'San Diego'", [], "new"),
    ("c03", CA, "single", "What is the highest average SAT reading score recorded for any school?",
     "SELECT MAX(AvgScrRead) FROM satscores WHERE rtype = 'S'", [], "new"),
    ("c04", CA, "single", "How many public high schools in Sacramento County are listed in the FRPM data (school type 'High Schools (Public)')?",
     "SELECT COUNT(*) FROM frpm WHERE `County Name` = 'Sacramento' AND `School Type` = 'High Schools (Public)'", [], "new"),
    ("c05", CA, "single", "What is the earliest opening date among schools in San Francisco County?",
     "SELECT MIN(OpenDate) FROM schools WHERE County = 'San Francisco'", [], "new"),
    ("c06", CA, "single", "How many distinct school districts are there in Kern County according to the schools directory?",
     "SELECT COUNT(DISTINCT District) FROM schools WHERE County = 'Kern'", [], "new"),
    ("c07", CA, "single", "Which 5 counties have the highest total FRPM count for K-12 students? List the county names.",
     "SELECT `County Name` FROM frpm GROUP BY `County Name` ORDER BY SUM(`FRPM Count (K-12)`) DESC LIMIT 5", [], "new"),
    ("c08", CA, "single", "Which city in Alameda County has the largest number of entries in the schools directory?",
     "SELECT City FROM schools WHERE County = 'Alameda' GROUP BY City ORDER BY COUNT(*) DESC LIMIT 1", [], "new"),
    ("c09", CA, "single", "What is the average number of SAT test takers per school in Los Angeles County?",
     "SELECT AVG(NumTstTakr) FROM satscores WHERE cname = 'Los Angeles' AND rtype = 'S'", [], "new"),
    # ---------------- california_schools: multi-table JOIN ----------------
    ("c10", CA, "join", "List the names and cities of schools whose average SAT math score is above 650.",
     f"SELECT T2.School, T2.City FROM satscores AS T1 INNER JOIN schools AS T2 ON {SAT_JOIN} WHERE T1.AvgScrMath > 650 AND T1.rtype = 'S'",
     [], "new"),
    ("c11", CA, "join", "What are the phone numbers of schools whose K-12 enrollment is over 4000?",
     "SELECT T2.Phone FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`Enrollment (K-12)` > 4000",
     [], "new"),
    ("c12", CA, "join", "What is the latitude and longitude of the school with the largest K-12 enrollment?",
     "SELECT T2.Latitude, T2.Longitude FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T1.`Enrollment (K-12)` DESC LIMIT 1",
     [], "new"),
    ("c13", CA, "join", "Which school has the highest average SAT writing score, and what is its phone number?",
     f"SELECT T2.School, T2.Phone FROM satscores AS T1 INNER JOIN schools AS T2 ON {SAT_JOIN} WHERE T1.rtype = 'S' ORDER BY T1.AvgScrWrite DESC LIMIT 1",
     [], "new"),
    ("c14", CA, "join", "What is the K-12 enrollment of the school with the highest average SAT math score?",
     f"SELECT T2.`Enrollment (K-12)` FROM satscores AS T1 INNER JOIN frpm AS T2 ON {SAT_JOIN} ORDER BY T1.AvgScrMath DESC LIMIT 1",
     [], "new"),
    ("c15", CA, "rule", "For schools in Alameda County that have SAT results, how many have an average math score above 600?",
     f"SELECT COUNT(*) FROM satscores AS T1 INNER JOIN schools AS T2 ON {SAT_JOIN} WHERE T2.County = 'Alameda' AND T2.School IS NOT NULL AND T1.AvgScrMath > 600",
     ["R8"], "new"),
    ("c16", CA, "join", "List the mailing cities of the schools whose free meal count for K-12 is above 3000.",
     "SELECT T2.MailCity FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`Free Meal Count (K-12)` > 3000",
     [], "new"),
    ("c17", CA, "join", "What is the street address of the school with the most SAT test takers scoring over 1500 in Los Angeles County?",
     f"SELECT T2.Street FROM satscores AS T1 INNER JOIN schools AS T2 ON {SAT_JOIN} WHERE T2.County = 'Los Angeles' AND T1.rtype = 'S' ORDER BY T1.NumGE1500 DESC LIMIT 1",
     [], "new"),
    ("c18", CA, "join", "Which school type (from the FRPM data) do the schools with an average SAT reading score above 600 belong to? List the school name and its school type.",
     f"SELECT T2.`School Name`, T2.`School Type` FROM satscores AS T1 INNER JOIN frpm AS T2 ON {SAT_JOIN} WHERE T1.AvgScrRead > 600",
     [], "new"),
    ("c19", CA, "join", "For the school with the highest FRPM count for K-12 students, what are its average SAT math score and its city?",
     f"SELECT T1.AvgScrMath, T3.City FROM satscores AS T1 INNER JOIN frpm AS T2 ON {SAT_JOIN} INNER JOIN schools AS T3 ON T2.CDSCode = T3.CDSCode ORDER BY T2.`FRPM Count (K-12)` DESC LIMIT 1",
     [], "new"),
    ("c20", CA, "join", "List the administrators' email addresses (first administrator) of schools whose average SAT math score is above 640.",
     f"SELECT T2.AdmEmail1 FROM satscores AS T1 INNER JOIN schools AS T2 ON {SAT_JOIN} WHERE T1.AvgScrMath > 640 AND T1.rtype = 'S'",
     [], "new"),
    ("c21", CA, "join", "What is the average K-12 enrollment of the schools that have more than 400 SAT test takers?",
     f"SELECT AVG(T2.`Enrollment (K-12)`) FROM satscores AS T1 INNER JOIN frpm AS T2 ON {SAT_JOIN} WHERE T1.NumTstTakr > 400",
     [], "new"),
    # ---------------- california_schools: business-rule questions ----------------
    ("c22", CA, "rule", "How many active schools are there in Santa Clara County?",
     "SELECT COUNT(*) FROM schools WHERE County = 'Santa Clara' AND StatusType = 'Active' AND School IS NOT NULL",
     ["R6"], "new"),
    ("c23", CA, "rule", "How many charter schools are there in Los Angeles County according to the FRPM data?",
     "SELECT COUNT(*) FROM frpm WHERE `County Name` = 'Los Angeles' AND `Charter School (Y/N)` = 1", ["R3"], "new"),
    ("c24", CA, "rule", "Which school has the highest SAT excellence rate among schools with at least 100 test takers?",
     "SELECT sname FROM satscores WHERE rtype = 'S' AND NumTstTakr >= 100 ORDER BY CAST(NumGE1500 AS REAL) / NumTstTakr DESC LIMIT 1",
     ["R7", "R8"], "new"),
    ("c25", CA, "rule", "How many high-poverty schools are there in Fresno County?",
     "SELECT COUNT(*) FROM frpm WHERE `County Name` = 'Fresno' AND `Percent (%) Eligible FRPM (K-12)` > 0.75", ["R9"], "new"),
    ("c26", CA, "rule", "List the top 3 schools by average SAT total score.",
     "SELECT sname FROM satscores WHERE rtype = 'S' ORDER BY (AvgScrRead + AvgScrMath + AvgScrWrite) DESC LIMIT 3",
     ["R10", "R8"], "new"),
    ("c27", CA, "rule", "What is the average eligible free rate for K-12 students across charter schools in San Diego County?",
     "SELECT AVG(`Free Meal Count (K-12)` / `Enrollment (K-12)`) FROM frpm WHERE `County Name` = 'San Diego' AND `Charter School (Y/N)` = 1",
     ["R1", "R3"], "new"),
    ("c28", CA, "rule", "How many exclusively virtual schools are currently active?",
     "SELECT COUNT(*) FROM schools WHERE Virtual = 'F' AND StatusType = 'Active'", ["R4", "R6"], "new"),
    ("c29", CA, "rule", "Which 3 schools had the most SAT test takers?",
     "SELECT sname FROM satscores WHERE rtype = 'S' ORDER BY NumTstTakr DESC LIMIT 3", ["R8"], "new"),
    ("c30", CA, "rule", "Among high-poverty schools, how many have an average SAT math score above 500?",
     f"SELECT COUNT(*) FROM satscores AS T1 INNER JOIN frpm AS T2 ON {SAT_JOIN} WHERE T2.`Percent (%) Eligible FRPM (K-12)` > 0.75 AND T1.AvgScrMath > 500",
     ["R9"], "new"),
    ("c31", CA, "rule", "How many active magnet schools are there in Fresno County?",
     "SELECT COUNT(*) FROM schools WHERE County = 'Fresno' AND Magnet = 1 AND StatusType = 'Active'", ["R5", "R6"], "new"),
    ("c32", CA, "rule", "What is the SAT excellence rate of the charter school with the most SAT test takers?",
     f"SELECT CAST(T1.NumGE1500 AS REAL) / T1.NumTstTakr FROM satscores AS T1 INNER JOIN frpm AS T2 ON {SAT_JOIN} WHERE T2.`Charter School (Y/N)` = 1 ORDER BY T1.NumTstTakr DESC LIMIT 1",
     ["R7", "R3"], "new"),
    ("c33", CA, "rule", "How many active charter schools in Alameda County are there? Use the schools directory status and the FRPM charter flag.",
     "SELECT COUNT(*) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`County Name` = 'Alameda' AND T1.`Charter School (Y/N)` = 1 AND T2.StatusType = 'Active'",
     ["R3", "R6"], "new"),
    # ---------------- bank_failures (DuckDB) ----------------
    ("f01", BF, "single", "How many bank failures are recorded in total?",
     "SELECT COUNT(*) FROM bank_failures", [], "new"),
    ("f02", BF, "single", "Which state had the most bank failures?",
     "SELECT State FROM bank_failures GROUP BY State ORDER BY COUNT(*) DESC LIMIT 1", [], "new"),
    ("f03", BF, "single", "In which year did the most bank failures happen?",
     "SELECT YEAR(Date) FROM bank_failures GROUP BY YEAR(Date) ORDER BY COUNT(*) DESC LIMIT 1", [], "new"),
    ("f04", BF, "single", "What is the name of the failed bank with the largest assets?",
     "SELECT Bank FROM bank_failures ORDER BY \"Assets ($mil.)\" DESC LIMIT 1", [], "new"),
    ("f05", BF, "single", "How many banks failed in California in 2009?",
     "SELECT COUNT(*) FROM bank_failures WHERE State = 'California' AND YEAR(Date) = 2009", [], "new"),
    ("f06", BF, "single", "Which acquirer took over the most failed banks (excluding cases with no acquirer)?",
     "SELECT \"Acquired by\" FROM bank_failures WHERE \"Acquired by\" <> 'None (insured depositors paid directly)' GROUP BY \"Acquired by\" ORDER BY COUNT(*) DESC LIMIT 1",
     [], "new"),
    ("f07", BF, "single", "What were the total assets (in $ millions) of banks that failed in Georgia?",
     "SELECT SUM(\"Assets ($mil.)\") FROM bank_failures WHERE State = 'Georgia'", [], "new"),
    ("f08", BF, "rule", "How many large bank failures are there?",
     "SELECT COUNT(*) FROM bank_failures WHERE \"Assets ($mil.)\" >= 10000", ["R11"], "new"),
    ("f09", BF, "rule", "How many bank failures happened during the financial crisis period?",
     "SELECT COUNT(*) FROM bank_failures WHERE Date BETWEEN DATE '2008-01-01' AND DATE '2011-12-31'", ["R12"], "new"),
    ("f10", BF, "rule", "How many failed banks were paid off directly to depositors?",
     "SELECT COUNT(*) FROM bank_failures WHERE \"Acquired by\" = 'None (insured depositors paid directly)'", ["R13"], "new"),
    ("f11", BF, "rule", "List the names of large banks that failed after the financial crisis period.",
     "SELECT Bank FROM bank_failures WHERE \"Assets ($mil.)\" >= 10000 AND Date > DATE '2011-12-31'", ["R11", "R12"], "new"),
    ("f12", BF, "rule", "Which state had the most bank failures during the financial crisis period, and how many?",
     "SELECT State, COUNT(*) FROM bank_failures WHERE Date BETWEEN DATE '2008-01-01' AND DATE '2011-12-31' GROUP BY State ORDER BY COUNT(*) DESC LIMIT 1",
     ["R12"], "new"),
]

# Held-out paraphrases for Step 6 (correction loop). Never used in the base set or to write knowledge.
PARAPHRASE = [
    ("p01", CA, "rule", "Give me the count of open schools in Santa Clara County.",
     "SELECT COUNT(*) FROM schools WHERE County = 'Santa Clara' AND StatusType = 'Active' AND School IS NOT NULL", ["R6"], "c22"),
    ("p02", CA, "rule", "In Fresno County, how many schools count as high-poverty?",
     "SELECT COUNT(*) FROM frpm WHERE `County Name` = 'Fresno' AND `Percent (%) Eligible FRPM (K-12)` > 0.75", ["R9"], "c25"),
    ("p03", CA, "rule", "Name the three schools with the best combined SAT score.",
     "SELECT sname FROM satscores WHERE rtype = 'S' ORDER BY (AvgScrRead + AvgScrMath + AvgScrWrite) DESC LIMIT 3", ["R10", "R8"], "c26"),
    ("p04", CA, "rule", "Which three schools had the largest number of students sitting the SAT?",
     "SELECT sname FROM satscores WHERE rtype = 'S' ORDER BY NumTstTakr DESC LIMIT 3", ["R8"], "c29"),
    ("p05", CA, "rule", "Count the high-poverty schools whose SAT math average exceeds 500.",
     f"SELECT COUNT(*) FROM satscores AS T1 INNER JOIN frpm AS T2 ON {SAT_JOIN} WHERE T2.`Percent (%) Eligible FRPM (K-12)` > 0.75 AND T1.AvgScrMath > 500", ["R9"], "c30"),
    ("p06", CA, "rule", "Among schools with 100 or more SAT takers, which one has the best excellence rate?",
     "SELECT sname FROM satscores WHERE rtype = 'S' AND NumTstTakr >= 100 ORDER BY CAST(NumGE1500 AS REAL) / NumTstTakr DESC LIMIT 1", ["R7", "R8"], "c24"),
    ("p07", BF, "rule", "Count the failures of large banks.",
     "SELECT COUNT(*) FROM bank_failures WHERE \"Assets ($mil.)\" >= 10000", ["R11"], "f08"),
    ("p08", BF, "rule", "How many banks failed during the financial crisis?",
     "SELECT COUNT(*) FROM bank_failures WHERE Date BETWEEN DATE '2008-01-01' AND DATE '2011-12-31'", ["R12"], "f09"),
    ("p09", BF, "rule", "How many bank failures ended with the FDIC paying insured depositors directly, with no acquiring bank?",
     "SELECT COUNT(*) FROM bank_failures WHERE \"Acquired by\" = 'None (insured depositors paid directly)'", ["R13"], "f10"),
    ("p10", CA, "rule", "How many charter schools does Los Angeles County have in the FRPM data?",
     "SELECT COUNT(*) FROM frpm WHERE `County Name` = 'Los Angeles' AND `Charter School (Y/N)` = 1", ["R3"], "c23"),
]


def run_gold(ds: str, sql: str):
    if ds == CA:
        con = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True)
        try:
            return con.execute(sql).fetchall()
        finally:
            con.close()
    con = duckdb.connect(str(DUCK), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def dump(items, path, kind):
    with open(path, "w") as f:
        for it in items:
            if kind == "base":
                id_, ds, tier, q, sql, rules, src = it
                rec = dict(id=id_, datasource=ds, tier=tier, question=q, gold_sql=sql, rules=rules, source=src)
            else:
                id_, ds, tier, q, sql, rules, orig = it
                rec = dict(id=id_, datasource=ds, tier=tier, question=q, gold_sql=sql, rules=rules, paraphrase_of=orig)
            rows = run_gold(ds, sql)
            rec["gold_rows"] = len(rows)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{id_:4} {tier:6} rows={len(rows):4}  {str(rows[:3])[:110]}")


if __name__ == "__main__":
    dump(BASE, OUT / "base.jsonl", "base")
    print("----")
    dump(PARAPHRASE, OUT / "paraphrase.jsonl", "para")
    from collections import Counter

    print(len(BASE), Counter(x[2] for x in BASE), Counter(x[1] for x in BASE))
