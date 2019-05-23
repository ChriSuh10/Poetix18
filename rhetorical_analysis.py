import subprocess
import shlex


def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout)


commands = '''
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Polypton: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_polypton";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Anadiplosis: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_anadiplosis";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Anaphora: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_anaphora";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Antimetabole: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_antimetabole";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Chiasmus: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/other/sonnet_chiasmus";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Conduplicatio: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_conduplicatio";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Epanalepsis: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_epanalepsis";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Epistrophe: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_epistrophe";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Epizeuxis: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_epizeuxis";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Ploce: {windowSize: 3} }" "sonnet_ploce";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Polysyndeton: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_polysyndeton";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Symploce: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/repitition/sonnet_symploce";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Oxymoron: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/other/sonnet_oxymoron";
mono rhetorica/bin/x64/Release/Rhetorica.exe "data/sonnets.txt" "{ Isocolon: {windowSize: 3} }" "/usr/project/xtmp/dp195/Poetix18/data/rhet/other/sonnet_isocolon";
'''

subprocess_cmd(commands)


