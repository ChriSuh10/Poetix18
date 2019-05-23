# Open a file: file
with open ('data/rhet/input.txt', 'r', encoding='iso-8859-1') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line

#content = [re.sub(r'[^\x00-\x7f]',r'', x.strip()).strip() + "\n" for x in content] 
#content = [x for x in content if(len(x.split(" ")) > 1)] 
#content = [x.replace(''''d''', '''ed''') for x in content]
#content = [x.replace(''''st''', '''ed''') for x in content]
content = [bytes(x,'utf-8').decode('utf-8','ignore') +"\n" for x in content] 
print(content)

file = open('data/rhet/sonnets.txt', 'w', encoding='utf-8')
file.writelines(["%s\n" % line.strip() for line in content])
