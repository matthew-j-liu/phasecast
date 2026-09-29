# Source downloads

`data/real_mineralogy/` is gitignored, so nothing on disk records which revision
of a USGS release the benchmark was built from. USGS revises data releases in
place (DS 764 is on version 1.1, August 2016), and a re-download that shifted a
value would otherwise change truth under a frozen benchmark silently.

This file is that record: one row per downloaded file, with the checksum of the
copy every number on record was computed against. Verify with

    cd data/real_mineralogy && shasum -a 256 -c ../../explorations/real_mineralogy/sources.sha256

A mismatch means the release was revised. That is not automatically a problem,
but it must be adjudicated rather than absorbed -- re-run `harvest.py` and the
generated tables, and check whether any scored number moved.

Two guards already exist inside the harvest and stay the first line of defence:
Appendix 10's percentages are derived from raw counts and asserted against the
workbook's own percent block, and the transcribed Pinatubo majors are asserted
to sum to 100 volatile-free.

## du Bray, USGS DS 764 -- Bodie Hills, California and Nevada

https://pubs.usgs.gov/ds/764/ (version 1.1, August 2016), downloaded 2026-07-01;
the pamphlet added 2026-08-06.

| file | bytes | sha256 |
|---|---|---|
| `dubray_ds764/Appendix1.xls` | 324608 | 7d5738f4f65671160271ce55a88485b6942fcd0b95f9c00f518beeb5f235771b |
| `dubray_ds764/Appendix2.xls` | 497152 | 233cd50f817972b358fe2c3cb07fb6340ad48e6952759dab01abfaeec462cdee |
| `dubray_ds764/Appendix3.xls` | 579072 | b185f5753e606da939ea7dcc0d299e8c84b5b7d491f9f0004d5decf591e698a4 |
| `dubray_ds764/DS764_pamphlet.pdf` | 6563200 | 5cbd96f3661823663da5d8b106d9be827ffac019a11c53d8117f39e5ac123b9a |

`DS764_pamphlet.pdf` is the **authoritative definition of every column** the
appendices carry, in its "Data Fields" section, and settled four questions the
files alone could not: `Cpx_Opx` is presence plus a dominance ordering rather
than a ratio, `ClrIndx` is *defined* as the sum of the mafic abundances rather
than an independent estimate, `Total_I_pct` is the pre-recalculation analytical
total, and the majors are recalculated to 100 percent volatile-free. It is a
document rather than data, so it is not read by any code -- but re-deriving
those four facts from the spreadsheets alone is guesswork, which is why it is
pinned here.

## du Bray, USGS DS 1099 -- Tonopah, Divide, Goldfield districts, Nevada

https://pubs.usgs.gov/ds/1099/ , downloaded 2026-07-01; the report added
2026-08-06.

| file | bytes | sha256 |
|---|---|---|
| `dubray_ds1099/Appendix01.xls` | 294912 | b46b1725e66f5f3472cba2771e44e1eed652efd22c8ddffe2f90e19bffe7bcbe |
| `dubray_ds1099/Appendix02.xlsx` | 242064 | 4b87cf153fcf22dac704560a269d49e55d43e4f3829bd5874232530192143dec |
| `dubray_ds1099/Appendix03.xlsx` | 479419 | 27715f05cd356eb66ee777bb7c00a8c8f6782c50f550c31b9904c12f07b96874 |
| `dubray_ds1099/Appendix10.xlsx` | 28246 | ef21dd0a8c6212e512d50da803582d0340f34694f8626ba39e89c7d07fa67eef |
| `dubray_ds1099/ds1099.pdf` | 9233552 | c2fe329bac6a03bfd494077d710f21bce3e097dd8673f62b81516527e896f177 |

`ds1099.pdf` carries the same "Data Fields" section as DS 764's pamphlet, plus
the `chem_src` key naming the seven publications its chemistry was compiled
from, going back to Ransome (1909).

## du Bray, USGS DS 942 -- 1.4 Ga A-type granitoid intrusions, conterminous US

https://pubs.usgs.gov/ds/0942/ (2015), downloaded 2026-09-29.

| file | bytes | sha256 |
|---|---|---|
| `dubray_ds942/1.4GaIntrsChemModData.xlsx` | 4638846 | e28024d16df86379310cc1eed5dc9adf9faef558b22ed1a6712d2762e5af721c |
| `dubray_ds942/ds942.pdf` | 6036439 | e1d3978fd1008aabc1f1a6d4adaa5597fb0903a5a407f36897c664f18adf6693 |

`ds942.pdf` defines every column in its "Data Fields" table 1: the modes are
point counts of stained slabs or thin sections in volume percent of the whole
rock, `TR` is below 0.5 volume percent, `Lithology` is capitalized when the
name is supported by modal data, and the majors are recalculated to 100
volatile-free with all Fe as FeO. Its "Geochemical and Modal Data" section
states the chemical screens that move a rock to the `Alt` tab and that
`PoorLocSamps` holds rocks without a usable location.

## du Bray, USGS PP 1793 -- Boulder batholith, Montana

https://pubs.usgs.gov/pp/1793/ (2012), downloaded 2026-09-29.

| file | bytes | sha256 |
|---|---|---|
| `dubray_pp1793/Appendix_1.xls` | 324096 | e4aa2990fb6c9c4929c1f0073fb15e67a99aac547c5669f478427f723f7fe68e |
| `dubray_pp1793/PP1793.pdf` | 9459438 | 9a59218d4b31f7b1b026cbbcb17217bb4f98d4021686b16c4b8479ffdc5f8cd2 |
| `dubray_pp1793/DS454.pdf` | 27699655 | 108f5cfc3497bb43a73c359ae7ddc39444f5a3f0f4fdf918055489e965318d96 |

`PP1793.pdf` has no "Data Fields" section; the appendix's own header defers to
du Bray and others (2009), USGS Data Series 454
(https://pubs.usgs.gov/ds/454/, downloaded 2026-09-29), whose "Data Fields"
section is the **authoritative definition of the appendix's columns**. It
states that the modes are point counts of thin sections or stained slabs in
volume percent summing to about 100, that `tr` is generally below 0.1 percent
(not DS 942's 0.5), that a blank mineral cell means the mineral is not present,
that `Acces(incl opq)` carries the opaque oxides for sources that did not count
them separately, and the chemical screens behind the altered rocks it removed
(including an analytical-total window of 95 to 103, wider than DS 942's 97 to
103).

## Pallister and others 2008 -- Mount St. Helens 2004-06 dome

https://pubs.usgs.gov/pp/1750/ (chapter 30) and https://pubs.usgs.gov/of/2008/1130/

| file | bytes | sha256 |
|---|---|---|
| `msh_pallister2008/pp2008-1750_chapter30.pdf` | 54994033 | b2f37344554deb78630efe498bf885ff0ea51e105cc73eb39e01334b14ee9043 |
| `msh_pallister2008/ch30.txt` | 325616 | a3e72b107ed715a2483a57c64e3fa54200db6bdbe6579b5dd3c0383cf521806b |
| `msh_pallister2008/of2008-1130_v1.1.xls` | 162816 | a7bf98fdd0ff5b83231fc68d3cc62a64b89938382c523a08ea8733563cebeb51 |
| `msh_pallister2008/Digital_Appendix_oxide_analyses.xls` | 236032 | 38614898b19a2b701ec8af7f537886a3e9945f4c295199b991817eed3fc22472 |
| `msh_pallister2008/Thornber_DATA_APPENDIX2_FINALv39.xls` | 404992 | a46e3762fc2c0fbaa85abeac4893340f9bfbe47435e4a8be7944d40f20d075c5 |

The MSH modes are transcribed by hand from chapter 30's Table 2 (a raster table
in the PDF), so the checksum pins the page the transcription was read off.

## Pallister and others 1996 -- Pinatubo 1991

https://pubs.usgs.gov/pinatubo/pallister/ . Modes and chemistry are transcribed
from `table1.gif` and `table7p*.gif`; the GIFs are not archived here, and the
transcription's own guard is the volatile-free sum assertion in `harvest.py`.
