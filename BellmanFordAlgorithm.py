###References: Stable matching template,networkx and stackoverflow

import glob
import sys
import matplotlib.pyplot as plt
import networkx as nx
import random
import os
import numpy as np


class Bellmanford1:
    def __init__(self):
        # Generate N number of Questions
        self.generate_nnumber_Of_questions()

    files = glob.glob('images/*')
    for f in files:
        os.remove(f)

    def computebellmanford(self, edgesWithWeights, n):
        # run the Bellman–Ford algorithm from every node
        returnValue = {}
        # for source in range(n):
        visited = [0] * n
        iterationStartPick = 0
        iterationEndPick = 0
        sourcePick = 0
        answers = []
        for i in range(n):
            source = random.randint(0, n - 1)
            if visited[source] != 1:
                visited[source] = 1
                iterationInfo = obj.bellmanFord(edgesWithWeights, source, n)
                negativeCycle = True
                for i in range(len(iterationInfo["cc"])):
                    if (iterationInfo["cc"][i] > 0):
                        negativeCycle = False
                        break
                if (sum(iterationInfo["cc"]) >= n - 1 and iterationInfo["cc"][0] != 7 and not negativeCycle):
                    sourcePick = source
                    iterationStartPickSuccess = False
                    for i in range(n - 3):
                        iterationStartPick = random.randint(0, n - 3)
                        if (iterationInfo["cc"][iterationStartPick] != 0):
                            iterationStartPickSuccess = True
                            break
                    iterationEndPick = iterationStartPick + random.randint(1, 2)
                    if iterationStartPickSuccess:
                        '''print("Debug-->Source", sourcePick, "Iteration Start Value", iterationStartPick, "Iteration End Value",
                              iterationEndPick)
                        print("Debug-->Change Count for Iteration Start Value", iterationInfo["cc"][iterationStartPick],
                              "Change count for Iteration End Value", iterationInfo["cc"][iterationEndPick])'''
                        for i in range(iterationStartPick, iterationEndPick + 1):
                            if (iterationInfo["cc"][i] > 0):
                                if (i == 0):
                                    answers.extend(iterationInfo[i])
                                else:
                                    answers.extend(iterationInfo[i][len(iterationInfo[i - 1]):])

                        returnValue['source'] = sourcePick
                        returnValue['iterationStart'] = iterationStartPick
                        returnValue['iterationEnd'] = iterationEndPick
                        returnValue['answers'] = set(answers)
                        break
        return returnValue

    def find_max_list_idx(self, list):
        list_len = [len(i) for i in list]
        return np.argmax(np.array(list_len))

    def constructGraph(self, vertices, edges, questionNo):
        filename = []
        F = nx.gnm_random_graph(vertices, edges, directed=True)
        G = nx.DiGraph()
        edge_list = [(u, v) for u, v in F.edges()]
        # print('Debug--->',edge_list)
        weighted_edge_list = [(u, v, random.randint(-4, 10)) for u, v in edge_list]
        # print('Debug--->',weighted_edge_list)
        negativeweights = list(filter(lambda x: x[2] < 0, weighted_edge_list))
        # print('Debug Test--->Negative weights::', negativeweights)
        # if no negative weights exist loop till altleast 1 negative weight graph is generated
        if not negativeweights:
            for _ in range(5):
                weighted_edge_list = [(u, v, random.randint(-4, 10)) for u, v in edge_list]
                # print('Debug Test--->New weighted_edge_list ',weighted_edge_list)
                negativeweights = list(filter(lambda x: x[2] < 0, weighted_edge_list))
                # print('Debug Test--->New negative weighted_edge_list ', negativeweights)
                if not negativeweights:
                    # print('Debug Test--->Reaches Continue')
                    continue
                else:
                    # print('Debug Test--->Breaks')
                    break

        for (u, v) in edge_list:
            G.add_weighted_edges_from(weighted_edge_list)

        G.edges(data=True)

        pos = nx.spring_layout(G, k=3, iterations=20)

        nx.draw(G, pos, with_labels=True, node_size=300, node_color='green')
        nx.draw_networkx_edge_labels(G, pos, font_size=9, edge_labels=nx.get_edge_attributes(G, 'weight'),
                                     bbox= dict(boxstyle="circle,pad=0.3",ec="black"), verticalalignment='baseline',label_pos=0.3)
        path = 'images'
        # Check whether the specified path exists or not
        isExist = os.path.exists(path)
        if isExist:
            if os.path.exists("images/BellmanFordGraph_" + str(questionNo) + ".png"):
                os.remove("images/BellmanFordGraph_" + str(questionNo) + ".png")
            plt.savefig("images/BellmanFordGraph_" + str(questionNo) + ".png", format="PNG")
            filename.append("BellmanFordGraph_" + str(questionNo) + ".png")
            plt.close()
            # plt.show()
        else:
            os.mkdir(path)
            plt.savefig("images/BellmanFordGraph_" + str(questionNo) + ".png", format="PNG")
            filename.append("BellmanFordGraph_" + str(questionNo) + ".png")
            plt.close()
            # plt.show()

        G.clear()
        return weighted_edge_list, filename, edge_list

    # Recursive function to print the path of a given vertex from source vertex
    def getPath(self, parent, vertex):
        if vertex < 0:
            return []
        return obj.getPath(parent, parent[vertex]) + [vertex]

    # Function to run the Bellman–Ford algorithm from a given source
    def bellmanFord(self, edgesWithWeight, source, n):
        # distance[] and parent[] stores the shortest path (least cost/path) info
        distance = [sys.maxsize] * n
        parent = [-1] * n
        vertexUpdated = []
        counter = 0
        changesCount = [0] * n
        iterationInfo = {}
        # Initially, all vertices except source vertex weight INFINITY and no parent
        distance[source] = 0
        # Randomizing the iteration
        # iterationChoice = [2,3]
        # randomIterationChoice = random.choice(iterationChoice)
        # print('randomIterationChoice: ',randomIterationChoice)
        # relaxation step (run V-1 times)
        for k in range(n - 1):
            # for k in range(randomIterationChoice):
            # newVertex = []
            # edge from `u` to `v` having weight `w`
            for (u, v, w) in edgesWithWeight:
                # if the distance to destination `v` can be shortened by taking edge (u, v)
                if distance[u] != sys.maxsize and distance[u] + w < distance[v]:
                    # update distance to the new lower value
                    distance[v] = distance[u] + w
                    # set v's parent as `u`
                    parent[v] = u
                    # newVertex.append(v)
                    changesCount[k] += 1
                    vertexUpdated.append(v)
                    for i in range(len(vertexUpdated)):
                        iterationInfo[k] = vertexUpdated.copy()
            # print('Vertices ', vertexUpdated, 'is updated in Iteration ', k)
        iterationInfo["cc"] = changesCount
        # print("Dictionary iterationInfo ::", iterationInfo)
        # run relaxation step once more for n'th time to check for negative-weight cycles
        for (u, v, w) in edgesWithWeight:  # edge from `u` to `v` having weight `w`
            # if the distance to destination `u` can be shortened by taking edge (u, v)
            if distance[u] != sys.maxsize and distance[u] + w < distance[v]:
                # print('Debug--->Negative-weight cycle is found!!')
                iterationInfo["cc"] = changesCount
                return iterationInfo

        for i in range(n):
            if i != source and distance[i] < sys.maxsize:
                '''print(Debug---->f'The distance of vertex {i} from vertex {source} is {distance[i]}. '
                      f'Its path is', obj.getPath(parent, i))'''

        return iterationInfo

    def printHeader(self):
        # print csv file header, erasing file content. Also put feedback list in the header.
        # uses drillheader and feedbacks variables
        # prepend every line with  the comment // and put 4 commas at the end to preserve the format.
        # the first part is the content of drillheader, then a line with the word "Feedbacks", then a numbered list of feedbacks (answer choice types)

        csvheader = '\n'.join(["//" + x + ",,,," for x in self.drillheader.split('\n') + \
                               ["Description: "] + self.drilldescription.split('\n') + \
                               ["Feedbacks:"] + [str(i) + ": " + s for i, s in enumerate(self.feedbacks)] + [""]])
        filename = "bellmanFord-template.csv"
        f = open(filename, "w", encoding="utf-8")
        f.write(csvheader)
        f.close()

    def generate_nnumber_Of_questions(self):
        ###################
        # general parameters
        ###################

        # number of questions: the first command-line argument
        if len(sys.argv) > 1:
            self.num_questions = int(sys.argv[1])
        else:
            self.num_questions = 10

        filename = "bellmanFord-template.csv"

        self.num_answers = 8  # number of answer options, usually at least 8, 10 is common, larger is OK for some questions.
        self.maxchecked = 4  # maximum number of options for which correct is to be selected (checked), in the range 1..num_answers
        self.minchecked = 1  # minimal number of options for which correct is to be selected (checked), in the range 1..maxchecked
        # note that D2L requires at least one option to be set as "checked" to submit a question
        self.points = self.num_answers  # leave as is.
        self.html = "HTML"  # whether the question and answers are in html (preferred) or plain text. If html, set html = "HTML"  if plain, set html = ""
        # Description of the drill, in particular answer choices/feedbacks types, for the output file header as plain text
        # start all subsequent lines with no indentation, and use triple double-quote marks at the start and end
        # if you need to use quotes in your text, write \"; same for \/ for /

        self.drillheader = """(Note: The images folder is assumed to be in the \/content\/<course path>\/ directory) 
Question Text is always a required field
Bellmanford: Identify all the vertices that are updated at least once between 2nd and 3rd iteration 
(including 2nd and 3rd iteration changes) of the bellman ford algorithm, 
considering vertex 1 as the source.: 
"""

        # include drill description in the csv file as well.
        # It will contain the text that will go into the D2L quiz intro/description,
        # in particular a solved example.
        if self.html == "HTML":
            self.drilldescription = """\"<html><p>This is a drill about BellmanFord, in particular, exploring to find the best path to a vertex from another vertex 
and in process to find the vertices that gets updated. 
To answer it, you need to select the vertices that get updated in the particular iteration.

For example, suppose that the question is as follows:  
Identify all the vertices that are updated at least once between 2nd and 3rd iteration (including 2nd and 3rd iteration changes) of the bellman ford algorithm, considering vertex 1 as the source
Edges: - (1,2) (1,3) (1,4) (2,5) (3,2) (3,5) (4,3) (4,6) (5,8) (6,7) (7,8)
Vertices:- 1 to 8
Options:
a) 1 <br>
b) 2 (correct) <br>
c) 3 <br>
d) 4 <br>
e) 5 (correct) <br>
f) 6 <br>
g) 7 <br>
h) 8 (correct) <br>
Note: - Graph indication:
1<----- (-2)-----9 ---->3
Here for bi-directional edges weights from 1 to 3 is 9 and 3 to 1 is -2. 
</p></html>\""""
        else:
            self.drilldescription = """This is a drill about BellmanFord, in particular, exploring to find the best path to a vertex from another vertex 
and in process to find the vertices that gets updated. 
To answer it, you need to select the vertices that get updated in the particular iteration.

For example, suppose that the question is as follows:  
Identify all the vertices that are updated at least once between 2nd and 3rd iteration (including 2nd and 3rd iteration changes) of the bellman ford algorithm, considering vertex 1 as the source
Edges: - (1,2) (1,3) (1,4) (2,5) (3,2) (3,5) (4,3) (4,6) (5,8) (6,7) (7,8)
Vertices:- 1 to 8
Options:
a) 1 <br>
b) 2 (correct) <br>
c) 3 <br>
d) 4 <br>
e) 5 (correct) <br>
f) 6 <br>
g) 7 <br>
h) 8 (correct) <br>
Note: - Graph indication:
1<----- (-2)-----9 ---->3
Here for bi-directional edges weights from 1 to 3 is 9 and 3 to 1 is -2. 
"""

        # an array of feedbacks corresponding to different types of answers (classified by this array index)
        # avoid using commas in feedbacks

        self.feedbacks = ["Wrong: This vertex has not been updated in this iteration",
                          "Correct: This vertex has been updated in this iteration"]

    def get_statement(self, sourceVertex, iterationStart, iterationEnd):
        # The statement needs to be one cell in a csv file, no newlines, \" at the beginning and end, and get it all into one line
        # (use \ at the end of lines to split text for readability), and put """ before and after.
        # If using html (like here), make sure to set html = True above

        statement = """\"<p>Identify all the vertices that are updated at least once between """ \
                    + str(iterationStart) + """ and """ + str(iterationEnd) + """ iteration (including """ + str(
            iterationStart) + """ and """ + str(iterationEnd) + """
iteration changes) of the bellman ford algorithm,considering vertex """ + str(sourceVertex) + """ as the source """ \
                                                                                              """ Hint: use this edgeList to solve :""" + str(
            edge_list) + """
              Note: Graph indication:
1<----- (-2)-----9 ---->3
Here for bi-directional edges weights from 1 to 3 is 9 and 3 to 1 is -2. 
</p>\""""
        return statement

    # text is a string of the answer choice
    # checked is a Boolean (True if correct answer is to check this option)
    # feedback_index is an int indexing into feedbacks array (in [0..length(feedbacks)-1])

    def print_answer(self, text, checked, feedback_index):
        if self.html == "HTML":
            answer_row = ",".join(
                ["Option", str(int(checked)), str(text), "HTML", self.feedbacks[feedback_index], "HTML"])
        else:
            answer_row = ",".join(["Option", str(int(checked)), str(text), self.feedbacks[feedback_index]])

        return answer_row

    def get_answers(self, answerList, vertices):
        # get an correct and incorrect vertices
        allVerticesList1 = [1, 2, 3, 4, 5, 6, 7, 8]
        allVerticesList = []
        allVerticesList.extend(range(vertices))
        # print('allVerticesList1', allVerticesList1)
        nonAnswersList = allVerticesList
        for v in answerList:
            nonAnswersList.remove(v)

        # print('answerlist', answerList)
        # print('nonanswerlist', nonAnswersList)
        # print('allVerticesList',allVerticesList)
        non_answers_sample = random.sample(nonAnswersList, len(nonAnswersList))
        # print('non_answers_sample', non_answers_sample)
        unstable_rows = [obj.print_answer(wrongAns, False, 0) for (wrongAns) in non_answers_sample]
        # print('unstable_rows', unstable_rows)
        num_correct = len(allVerticesList1) - len(nonAnswersList)
        # print('num_correct', num_correct)
        # select arbitrary stable pairs
        answers_sample = random.sample(answerList, num_correct)
        stable_rows = [obj.print_answer(correctAns, True, 1) for (correctAns) in answers_sample]
        ans_list = unstable_rows + stable_rows
        # randomize the order of the pairs
        random.shuffle(ans_list)
        # return a string containing all answers in csv format
        return "\n".join(ans_list)

    def print_image_question(self, num, statement, answers, image):
        question = """,,,,
//variant """ + str(num) + """,,,,
NewQuestion,MS,,,
Title,,,,
QuestionText,""" + statement + "," + self.html + """,,
Points,""" + str(self.points) + """,,,
Difficulty,1,,,
Image,images/""" + image + """,,,
Scoring,RightAnswers,,, 
""" + answers + """
"""
        return question


if __name__ == '__main__':
    obj = Bellmanford1()
    obj.printHeader()

    # Add vertices and edges as a parameter
    vertices = 8
    edges = 10
    filename = "bellmanFord-template.csv"
    f = open(filename, "a", encoding="utf-8")
    noOfQuestions = obj.num_questions
    for i in range(noOfQuestions):
        while (True):
            weighted_edge_list, filename, edge_list = obj.constructGraph(vertices, edges, i + 1)
            resultSet = obj.computebellmanford(weighted_edge_list, vertices)

            if len(resultSet) == 4:
                print(resultSet)
                break
        sourceVertex = resultSet.get('source')
        iterationStart = resultSet.get("iterationStart")
        iterationEnd = resultSet.get("iterationEnd")
        answers = resultSet.get("answers")
        statement = obj.get_statement(sourceVertex, iterationStart, iterationEnd)
        # print('Statement is: ',statement)
        # print('vertices is',vertices)
        answers = obj.get_answers(answers, vertices)
        # print('Answer is: ', answers)
        print('filename++++++++++++++++++++++', filename)
        f.write(obj.print_image_question(i, statement, answers, filename[0]))
    f.close()

