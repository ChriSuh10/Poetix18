# Open a file: file
import argparse
import os


#os.getcwd()
parser = argparse.ArgumentParser(description='Do rhetorical analysis')

parser.add_argument('input', help='''Input file name (located in data/) e.g. sonnets.txt''')


args = parser.parse_args()

local_path_input = 'data/' + args.input
abs_path_input = os.path.abspath(local_path_input)
filename, file_extension = os.path.splitext(abs_path_input)
corpus, extension = os.path.splitext(args.input)

local_path_output = filename + '_input' + '.txt'


with open (abs_path_input, 'r', encoding='iso-8859-1') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line

#content = [re.sub(r'[^\x00-\x7f]',r'', x.strip()).strip() + "\n" for x in content] 
#content = [x for x in content if(len(x.split(" ")) > 1)] 
#content = [x.replace(''''d''', '''ed''') for x in content]
#content = [x.replace(''''st''', '''ed''') for x in content]
content = [bytes(x,'utf-8').decode('utf-8','ignore') +"\n" for x in content] 

file = open(local_path_output, 'w+', encoding='utf-8')
one_word = ""
for line in content:
	if(len(line.strip().split()) == 1):
		one_word = one_word + line.strip()
		#print(one_word)
	else:
		file.write("%s\n" % line.strip())
file.close()

abs_path_output = os.path.abspath(local_path_output)

rhet_file = os.path.basename(abs_path_output)

import subprocess
import os
import py_files.Rhetoric as Rhetoric

def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout)

pwd_dir = os.path.dirname(os.path.realpath(__file__))
output_folder_rep = "%s/data/rhet/repitition" % (pwd_dir)
config = 'Release'
platform = 'x64'

commands1 = '''
xbuild rhetorica/Rhetorica.sln /p:Configuration=%s /p:Platform=%s;
''' % (config, platform)
print(commands1)
subprocess_cmd(commands1)

commandsR = []
for e in Rhetoric.RHETORICAL_FIGURES:
    command = '''mono rhetorica/bin/%s/%s/Rhetorica.exe "%s/" "%s" "{ %s: {windowSize: 3} }" "%s/%s_%s";''' % (platform, config, pwd_dir, rhet_file, e.name.lower().capitalize(), output_folder_rep, corpus, e.name.lower())
    print(command)
    subprocess_cmd(command)




# commands3 = ['''mono rhetorica/bin/%s/%s/Rhetorica.exe "%s/" "%s" "{ Anadiplosis: {windowSize: 3} }" "%s/%s_anadiplosis;"''' % (config, platform, pwd_dir, rhet_file, output_folder, corpus),
# '''mono  rhetorica/bin/%s/%s/Rhetorica.exe "%s/" "%s" "{ Anaphora: {windowSize: 3} }" "%s/%s_anaphora;"''' % (config, platform, pwd_dir, rhet_file, output_folder, corpus),
# '''mono  rhetorica/bin/%s/%s/Rhetorica.exe "%s/" "%s" "{ Antimetabole: {windowSize: 3} }" "%s/%s_antimetabole;"''' % (config, platform, pwd_dir, rhet_file, output_folder, corpus),
# '''mono  rhetorica/bin/%s/%s/Rhetorica.exe "%s/" "%s" "{ Conduplicatio: {windowSize: 3} }" "%s/%s_conduplicatio;"''' % (config, platform, pwd_dir, rhet_file, output_folder, corpus),
# '''mono  rhetorica/bin/%s/%s/Rhetorica.exe "%s/" "%s" "{ Epanalepsis: {windowSize: 3} }" "%s/%s_epanalepsis;"''' % (config, platform, pwd_dir, rhet_file, output_folder, corpus),
# '''mono  rhetorica/bin/%s/%s/Rhetorica.exe "%s/" "%s" "{ Epistrophe: {windowSize: 3} }" "%s/%s_epistrophe;"''' % (config, platform, pwd_dir, rhet_file, output_folder, corpus),
# '''mono  rhetorica/bin/%s/%s/Rhetorica.exe "%s/" "%s" "{ Epizeuxis: {windowSize: 3} }" "%s/%s_epizeuxis;"''' % (config, platform, pwd_dir, rhet_file, output_folder, corpus),
# '''mono  rhetorica/bin/%s/%s/Rhetorica.exe "%s/" "%s" "{ Ploce: {windowSize: 3} }" "%s/%s_ploce;"''' % (config, platform, pwd_dir, rhet_file, output_folder, corpus),
# '''mono  rhetorica/bin/%s/%s/Rhetorica.exe "%s/" "%s" "{ Polysyndeton: {windowSize: 3} }" "%s/%s_polysyndeton;"''' % (config, platform, pwd_dir, rhet_file, output_folder, corpus),
# '''mono  rhetorica/bin/%s/%s/Rhetorica.exe "%s/" "%s" "{ Symploce: {windowSize: 3} }" "%s/%s_symploce;"''' % (config, platform, pwd_dir, rhet_file, output_folder, corpus)],

#commandsRs = "\n".join(commandsR)
#print(commandsRs)

#subprocess_cmd(commands1)
#subprocess_cmd(commands2)
#subprocess_cmd(commandsR)



