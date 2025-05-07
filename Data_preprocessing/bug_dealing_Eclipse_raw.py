#!/usr/lib/env python3
# -*- code: latin-1 -*-

# import
import os
import pandas as pd
import csv
from dateutil import parser
from pandas.tseries.offsets import Hour
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from openai import OpenAI
import json


# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# 数据位置
data_dir = "../../data/data_by_ocean/Eclipse_raw/"
bug_list_dir = "buglist/"
bug_description_dir = "description/"
bug_history_dir = "bughistory_raw/"

def textProcessing(text_0, text_1):


# ------------------------------- Dictionary -----------------------------------
    translation_dict = {
        "WTP": "Web Tools Platform",
        "AJDT": "AspectJ Development Tools",
        "CDT": "C/C++ Development Tooling",
        "EMF": "Eclipse Modeling Framework",
        "GEF": "Graphical Editing Framework",
        "GMP": "Graphical Modeling Project",
        "GMF": "Graphical Modeling Framework",
        "JDT": "Java Development Tools",
        "JSDT": "JavaScript Development Tools",
        "PDT": "PHP Development Tools",
        "SWT": "Standard Widget Toolkit",
        "TPTP": "Test and Performance Tools Platform",
        "BIRT": "Business Intelligence and Reporting Tools",
        "OCL": "Object Constraint Language",
        "EPP": "Eclipse Packaging Project",
        "RAP": "Remote Application Platform",
        "JET": "Java Emitter Templates",
        "M2T": "Model-to-Text",
        "EMFT": "Eclipse Modeling Framework Technology",
        "MDT": "Model Development Tools",
        "ATL": "Atlas Transformation Language",
        "ETL": "Extract, Transform, Load",
        "GIT": "Version Control System",
        "ECF": "Eclipse Communication Framework",
        "MOXy": "Mapping Objects to XML",
        "MTJ": "Mobile Tools for Java",
        "JPA": "Java Persistence API",
        "RCP": "Rich Client Platform",
        "XML": "Extensible Markup Language",
        "XWT": "XML Window Toolkit",
        "XSD": "XML Schema Definition",
        "LDT": "Lua Development Tools",
        "SDK": "Software Development Kit",
        "OTDT": "Object Teams Development Tooling",
        "GEM": "Graphical Eclipse Modeling",
        "CVS": "Concurrent Versions System",
        "EGit": "Eclipse Git Integration",
        "JGit": "Java Git Library",
        "Tycho": "Maven Build System for Eclipse",
        "OSGi": "Open Services Gateway Initiative",
        "RCP": "Rich Client Platform",
        "EPF": "Eclipse Process Framework",
        "STS": "Spring Tool Suite",
        "JSF": "Java Server Faces",
        "SCA": "Service Component Architecture",
        "GWT": "Google Web Toolkit",
        "RIA": "Rich Internet Application",
        "DTP": "Data Tools Platform",
        "H2": "Lightweight Java SQL Database",
        "RMF": "Requirements Modeling Framework",
        "TCF": "Target Communication Framework",
        "QVT": "Query/View/Transformation",
        "OTJ": "Object Teams for Java",
        "VE": "Visual Editor",
        "UML": "Unified Modeling Language",
        "RIA": "Rich Internet Applications",
        "JWT": "JSON Web Token",
        "API": "Application Programming Interface",
        "SDK": "Software Development Kit",
        "CLI": "Command Line Interface",
        "UI": "User Interface",
        "IDE": "Integrated Development Environment",
        "DBWS": "Database Web Services",
        "AJBrowser": "AspectJ Browser",
        "XMI": "XML Metadata Interchange",
        "Xtext": "DSL Framework for Eclipse",
        "DSL": "Domain Specific Language",
        "SWTBot": "SWT UI Testing Framework",
        "JET": "Java Emitter Templates",
        "MTJ": "Mobile Tools for Java",
        "XWT": "XML Window Toolkit",
        "Hudson": "Continuous Integration Server",
        "JIRA": "Bug Tracking System",
        "Tcl": "Tool Command Language",
        "JAXB": "Java Architecture for XML Binding",
        "XUL": "XML User Interface Language",
        "RIA": "Rich Internet Application",
        "JWT": "JSON Web Token",
        "RPM": "Red Hat Package Manager",
        "DLTK": "Dynamic Languages Toolkit",
        "EMFT": "Eclipse Modeling Framework Technology",
        "GEM": "Graphical Eclipse Modeling",
        "VE": "Visual Editor",
        "OCL": "Object Constraint Language",
        "SCA": "Service Component Architecture",
        "LTTng": "Linux Trace Toolkit Next Generation",
        "OTMvn": "Object Teams Maven Plugin",
    }

    def translate_term(term):
        """
        Function to translate software-related abbreviations.
        If a translation exists in the dictionary, return it.
        Otherwise, return the original term.
        """
        return translation_dict.get(term, term)

# ------------------------------------------------------------------------------


# # ------------------------------- LLM -----------------------------------

#     client = OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1/")

#     text_00 = text_0[0]
#     text_01 = text_0[1]

#     print("text_00 --->" + text_00)
#     print("text_01 --->" + text_01)

#     response = client.chat.completions.create(
#                 model="qwen2.5-coder:32b",
#                 messages=[
#                     {"role": "system", 
#                     "content": """
#                  Please convert any "abbreviated terms" that may be contained in the following text content related to the field of "Computer Science and Applications - Software Design" to "original expressions," and return the information content strictly according to the following "JSON" format:
# {
# "content":{
# "text_0":""<If there is no "original expression," then output original input content>,
# "text_1":""<If there is no "original expression," then output original input content>,
# }
# }
# Example input: “{text_00:PDE, text_01:UI}”,
# Example output: “
# {
# "content":{
# "text_0":"Partial Differential Equation",
# "text_1":"User Interface"
# }
# }
# Note: Cannot be output in the following way:
# ```json
# {
# "content":{
# "text_0":"Partial Differential Equation",
# "text_1":"User Interface"
# }
# }
# ```

#                     """},
#                     {"role":"user", "content":"{" + "text_00:" + f"{text_00}" + "," + "text_01:" + f"{text_01}" + "}"}
#                 ],
#                 stream=False
#             )
            
#     response_content = response.choices[0].message.content
            
#     # Remove "json" prefix if exists
#     if response_content.lower().startswith('json'):
#         response_content = response_content[response_content.find('{'):]

#     print(response_content)

#     response_json = json.loads(response_content)
#     response_content = response_json["content"]
    
#     print(response_content)

# ------------------------------------------------------
        
    
                        
    # Tokenize
    tokens = word_tokenize(text_1.lower())
                        
    # Remove stop words and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
                        
    # Stem words
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
                        
    # Join back into string

    textProcessing_Test = [translate_term(text_0[0]), translate_term(text_0[1]), tokens]



    return textProcessing_Test
    
    
    

# 如果没有处理，则读入源数据，进行处理
if not os.path.exists(data_dir + 'raw/bug_raw.csv'):
    # 将所有数据都放入bug_raw.csv文件，分割符为@@,,@@，很无奈，处理的文本包括太多的字符了
    with open(data_dir + 'raw/bug_raw.csv', 'a', encoding='latin-1') as f:
        f.write('when@@,,@@who@@,,@@product@@,,@@component@@,,@@summary_description_comment\n')
    # 分别处理20个文件里面的bug
    for i in range(20):
        j = i + 1
        # 读取 bug list 的文件，获取bug id和summary
        bug_list_record = pd.read_csv(
            data_dir + bug_list_dir + 'bugs' + str(j) + '.csv', encoding='latin-1')
        # 针对bug list里面的每一个bug，我们将读取其描述文件和bug history文件
        for bug in bug_list_record.values:
            print(bug)
            print(len(bug))
            print('description: ' + data_dir + bug_description_dir +
                  'bugs' + str(i + 1) + '/' + str(bug[0]) + '.txt')
            # 读取bug的description文件，简单的将每行文本提取出来，并去掉多余空格，组合起来便是一个描述性文件
            with open(data_dir + bug_description_dir + 'bugs' + str(i + 1) + '/' + str(bug[0]) + '.txt', 'r',
                      encoding='latin-1') as description_f:
                bug_description_record = " ".join(
                    map(str.strip, description_f.readlines()))
                
            # with open(data_dir + bug_description_dir + 'bugs' + str(i + 1) + '/' + str(bug[0]) + '.txt', 'r',
            #           encoding='latin-1') as description_D:
                

            # 预处理history文件，原始文件格式不统一，使用pd.read_csv方法总是失败
            # 故此处只处理前五列的数据，以方便下文处理
            if not os.path.exists(data_dir + bug_history_dir + 'bugs' + str(i + 1) + '/' + str(bug[0]) + '.delt.csv'):
                with open(data_dir + bug_history_dir + 'bugs' + str(i + 1) + '/' + str(bug[0]) + '.csv', 'r',
                          encoding='latin-1') as history_f:
                    data = [line.strip().split(',')[:5]
                            for line in history_f.readlines()]
                    with open(data_dir + bug_history_dir + 'bugs' + str(i + 1) + '/' + str(bug[0]) + '.delt.csv', 'w',
                              encoding='latin-1') as write_f:
                        for line in data:
                            write_f.write(",".join(line) + '\n')

            print('history: ' + data_dir + bug_history_dir + 'bugs' +
                  str(i + 1) + '/' + str(bug[0]) + '.delt.csv')
            # 读取history文件
            bug_history_record = pd.read_csv(
                data_dir + bug_history_dir + 'bugs' +
                str(i + 1) + '/' + str(bug[0]) + '.delt.csv',
                quoting=csv.QUOTE_NONE, encoding='latin-1')
            # 将history按时间排序，提取 Added和What数据
            bug_history = bug_history_record.sort_values(
                ['When'], ascending=False)
            bug_history_record = bug_history[
                (bug_history[' Added'] == 'FIXED') & (bug_history[' What'] == 'Resolution')]
            # 如果成功提取到history数据，则将问问分别
            # 写入bug_summary_raw,bug_description_raw,bug_list_raw,bug_raw文件
            # 其中bug_raw文件是汇总文件
            if len(bug_history_record.values):
                with open(data_dir + 'raw/bug_summary_raw.csv', 'a', encoding='latin-1') as f:
                    f.write(str(bug[8]) + '\n')
                with open(data_dir + 'raw/bug_description_raw.csv', 'a', encoding='latin-1') as f:
                    f.write(bug_description_record + '\n')
                with open(data_dir + 'raw/bug_list_raw.csv', 'a', encoding='latin-1') as f:
                    f.write(bug_history_record['When'].values[0] +
                            ',' + bug_history_record[' Who'].values[0] + '\n')
                with open(data_dir + 'raw/bug_raw.csv', 'a', encoding='latin-1') as f:



                    #借助 大模型 调用 优化 bug[2]:所属产品 and bug[3]:所属组件 的 专业用词，例： 使用 "user interface" 替换 "UI" . 
                    #处理 描述 与 评论 字段严谨性.  --- 标记
                    #将每个缺陷的标题、描述以及评论整合成一个文本,并对其进行分词、去除停用词、去除数字与非字母、提取词干等操作．此外,将所属产品和所属组件信息与上述预处理后的文本进行合并,形成最终的缺陷报告文本内容．

                    text_0 = [str(bug[2]), str(bug[3])]

                    text_1 = str(str(bug[8]) + bug_description_record)

                    ProcessedText = textProcessing(text_0, text_1)

                    
                    line = bug_history_record['When'].values[0] + '@@,,@@' + bug_history_record[' Who'].values[0] + '@@,,@@' + str(ProcessedText[0]) + '@@,,@@' + str(ProcessedText[1]) + '@@,,@@' + str(ProcessedText[2])
                    print(line)
                    f.write(line + '\n')



else:
    def date_parse(time):
        # print(time)
        tz = time.split(' ')[2]
        dt = parser.parse(time)
        if tz == 'EDT':
            return dt + Hour(12)
        elif tz == 'EST':
            return dt + Hour(13)
        elif tz == 'PDT':
            return dt + Hour(15)
        elif tz == 'PST':
            return dt + Hour(16)
        else:
            print('缺少时区：%s' % tz)

    # 读取预处理文件
    bug_raw = pd.read_csv(data_dir + 'raw/bug_raw.csv',
                          sep='@@,,@@', engine='python', encoding='latin-1', parse_dates=[0], date_parser=date_parse)
    # 按照时间将数据进行排序
    bug_sorted_raw = bug_raw.sort_values('when')
    # 将排序好的数据写入磁盘备份
    bug_sorted_raw.to_csv(data_dir + 'raw/sorted_summary_description_comment.csv', columns=['summary_description_comment'],
                          header=False, index=False)
    bug_sorted_raw.to_csv(data_dir + 'raw/sorted_date_who.csv',
                          columns=['when', 'who'], index=False)
    bug_sorted_raw.to_csv(data_dir + 'raw/sorted_who.csv',
                          columns=['who'], index=False)
    # 将数据分成11份
    bug_len_0 = len(bug_sorted_raw)
    bug_part_size = int((bug_len_0 - 1) / 11) + 1


    for i in range(11):
        begin_index = i * bug_part_size
        end_index = min((i + 1) * bug_part_size, bug_len_0 - 1)
        bug_parted_0 = bug_sorted_raw.iloc[begin_index:end_index]

        bug_parted_0.to_csv(data_dir + "Segmented_content/Complete_data/" + str(i) + '.csv',
                          header=False, index=False)
        
        bug_parted_0.to_csv(data_dir + "Segmented_content/when/" + str(i) + '.csv', columns = ['when'],
                          header=False, index=False)

        bug_parted_0.to_csv(data_dir + "Segmented_content/who/" + str(i) + '.csv', columns = ['who'],
                                header=False, index=False)
        
        bug_parted_0.to_csv(data_dir + "Segmented_content/when_who/" + str(i) + '.csv', columns = ['when', 'who'],
                          header=False, index=False)


