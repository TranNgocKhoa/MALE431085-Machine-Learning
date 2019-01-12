import math
import numpy as np
class C45:

	"""Creates a decision tree with C4.5 algorithm"""
	def __init__(self, pathToData,pathToNames):
		self.filePathToData = pathToData
		self.filePathToNames = pathToNames
		self.data = []
		self.classes = []
		self.numAttributes = -1 
		self.attrValues = {}
		self.attributes = []
		self.tree = None


	def fetchData(self):
		'''
		Đọc file, các tên cột ở mỗi hàng cách nhau là dấu phẩy
		Các dòng tiếp theo, gồm thuộc tính, giá trị, cách nhau dấu ':'

		'''
		with open(self.filePathToNames, "r") as file:
			classes = file.readline()
			self.classes = [x.strip() for x in classes.split(",")]
			#add attributes
			for line in file:
				[attribute, values] = [x.strip() for x in line.split(":")]
				# print("values: ", values.split(","))
				values = [x.strip() for x in values.split(",")]
				# Dictionary chứa attribute <-> value Eg: length - continious
				self.attrValues[attribute] = values
		# Số lượng attribute
		self.numAttributes = len(self.attrValues.keys())
		# Danh sách Attribute
		self.attributes = list(self.attrValues.keys())
		# Đọc data
		with open(self.filePathToData, "r") as file:
			for line in file:
				row = [x.strip() for x in line.split(",")]
				if row != [] or row != [""]:
					self.data.append(row)

	'''
	Hàm này tiền xử lý data
	Nếu attribute không rời rạc thì cast nó thành float
	'''
	def preprocessData(self):
		for index,row in enumerate(self.data):
			for attr_index in range(self.numAttributes):
				if(not self.isAttrDiscrete(self.attributes[attr_index])):
					self.data[index][attr_index] = float(self.data[index][attr_index])

	def printTree(self):
		self.printNode(self.tree)

	def printNode(self, node, indent=""):
		if not node.isLeaf:
			if node.threshold is None:
				#discrete
				for index,child in enumerate(node.children):
					if child.isLeaf:
						print(indent + node.label + " = " + attributes[index] + " : " + child.label)
					else:
						print(indent + node.label + " = " + attributes[index] + " : ")
						self.printNode(child, indent + "	")
			else:
				#numerical
				leftChild = node.children[0]
				rightChild = node.children[1]
				if leftChild.isLeaf:
					print(indent + node.label + " <= " + str(node.threshold) + " : " + leftChild.label)
				else:
					print(indent + node.label + " <= " + str(node.threshold)+" : ")
					self.printNode(leftChild, indent + "	")

				if rightChild.isLeaf:
					print(indent + node.label + " > " + str(node.threshold) + " : " + rightChild.label)
				else:
					print(indent + node.label + " > " + str(node.threshold) + " : ")
					self.printNode(rightChild , indent + "	")


	def predict(self, list_vals):
		y = [self.predictValue(list_val, self.tree) for list_val in list_vals]
		return np.asarray(y, dtype=np.str)

	def predictValue(self, list_attribute_val, node):
		if not node.isLeaf:
			if node.threshold is None:
				i = self.attributes.index(node.label)
				for child in node.children:
					if list_attribute_val[i] == child.label:
						if child.isLeaf:
							return child.label
						else:
							return self.predictValue(list_attribute_val, child)

			else:
				#numerical
				i = self.attributes.index(node.label)
				leftChild = node.children[0]
				rightChild = node.children[1]
				if(list_attribute_val[i] > node.threshold):
					if(rightChild.isLeaf): 
						return rightChild.label
					else:
						return self.predictValue(list_attribute_val, rightChild)
				else:
					if leftChild.isLeaf:
						return leftChild.label
					else:
						return self.predictValue(list_attribute_val, leftChild)

	def generateTree(self):
		self.tree = self.recursiveGenerateTree(self.data, self.attributes)

	def recursiveGenerateTree(self, curData, curAttributes):
		allSame = self.allSameClass(curData)

		'''
		Nếu nhận vào dữ liệu là rỗng, trả về Node(isLeaf = True, label = "Fail", threshold = None)
		Nếu tất cả các dòng cùng một class, trả về Node(isLeaf = True, label = "True", threshold = "None")
		Nếu danh sách các Attribute rỗng, trả về có label là Class chiếm ưu thế trong curData
		Ngược lại...
		'''
		if len(curData) == 0:
			#Fail
			return Node(True, "Fail", None)
		elif allSame is not False:
			#return a node with that class
			return Node(True, allSame, None)
		elif len(curAttributes) == 0:
			#return a node with the majority class
			majClass = self.getMajClass(curData)
			return Node(True, majClass, None)
	
		else:
			'''
			Ngược lại, Tách attribute
			'''
			(best,best_threshold,splitted) = self.splitAttribute(curData, curAttributes)
			remainingAttributes = curAttributes[:]
			remainingAttributes.remove(best)
			node = Node(False, best, best_threshold)
			# print(node.label)
			# print(node.threshold)
			node.children = [self.recursiveGenerateTree(subset, remainingAttributes) for subset in splitted]
			return node

	def getMajClass(self, curData):
		freq = [0]*len(self.classes)
		for row in curData:
			index = self.classes.index(row[-1])
			freq[index] += 1
		maxInd = freq.index(max(freq))
		return self.classes[maxInd]


	'''
	Hàm này kiểm tra các row có cùng class với nhau ko
	row[-1] là lấy cột cuối cùng
	'''
	def allSameClass(self, data):
		for row in data:
			if row[-1] != data[0][-1]:
				return False
		return data[0][-1]

	def isAttrDiscrete(self, attribute):
		if attribute not in self.attributes:
			raise ValueError("Attribute not listed")
		elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == "continuous":
			return False
		else:
			return True

	'''
	Nếu thuộc tính liên tục:
	Trả về tên thuộc tính
	Khoảng phân chia
	Danh sách lớn hơn khoảng phân chia, danh sách nhỏ hơn khoảng phân chia
	Nếu thuộc tính không liên tục:
	Trả về thuộc tính tốt nhất
	Trả về danh sách value của thuộc tính
	Khoảng phân chia là None
	'''
	def splitAttribute(self, curData, curAttributes):
		splitted = []
		maxEnt = -1*float("inf")
		best_attribute = -1
		#None for discrete attributes, threshold value for continuous attributes
		best_threshold = None
		for attribute in curAttributes:
			indexOfAttribute = self.attributes.index(attribute)
			if self.isAttrDiscrete(attribute):
				#split curData into n-subsets, where n is the number of 
				#different values of attribute i. Choose the attribute with
				#the max gain
				'''
				Tách curData thành n-subsets, n là số lượng values khác nhau của attribute i
				Chọn attribute mà có max gain.
				'''
				valuesForAttribute = self.attrValues[attribute]
				subsets = [[] for a in valuesForAttribute]
				for row in curData:
					for index in range(len(valuesForAttribute)):
						if row[i] == valuesForAttribute[index]:
							subsets[index].append(row)
							break
				e = gain(curData, subsets)
				if e > maxEnt:
					maxEnt = e
					splitted = subsets
					best_attribute = attribute
					best_threshold = None
			else:
				#sort the data according to the column.Then try all 
				#possible adjacent pairs. Choose the one that 
				#yields maximum gain
				'''
				Sắp xếp data dựa trên cột.
				Thử tất cả các cặp liền kề
				Chọn cặp có max gain

				'''
				curData.sort(key = lambda x: x[indexOfAttribute])
				for j in range(0, len(curData) - 1):
					# Nếu dòng j tại cột attribute i khác dòng tiếp theo
					# Thì threshold (khoảng) = trung bình cộng
					if curData[j][indexOfAttribute] != curData[j+1][indexOfAttribute]:
						threshold = (curData[j][indexOfAttribute] + curData[j+1][indexOfAttribute]) / 2
						less = []
						greater = []
						for row in curData:
							if(row[indexOfAttribute] > threshold):
								greater.append(row)
							else:
								less.append(row)
						e = self.gain(curData, [less, greater])
						if e >= maxEnt:
							splitted = [less, greater]
							maxEnt = e
							best_attribute = attribute
							best_threshold = threshold
		return (best_attribute,best_threshold,splitted)

	def gain(self,unionSet, subsets):
		#input : data and disjoint subsets of it
		#output : information gain ratio
		S = len(unionSet)
		#calculate impurity before split
		impurityBeforeSplit = self.entropy(unionSet)
		#calculate impurity after split
		weights = [len(subset)/S for subset in subsets]
		impurityAfterSplit = 0
		for i in range(len(subsets)):
			impurityAfterSplit += weights[i]*self.entropy(subsets[i])
		#calculate total gain ratio
		totalGain = (impurityBeforeSplit - impurityAfterSplit) / impurityBeforeSplit
		return totalGain

	def entropy(self, dataSet):
		S = len(dataSet)
		if S == 0:
			return 0
		num_classes = [0 for i in self.classes]
		for row in dataSet:
			classIndex = list(self.classes).index(row[-1])
			num_classes[classIndex] += 1
		num_classes = [x/S for x in num_classes]
		ent = 0
		for num in num_classes:
			ent += num*self.log(num)
		return ent*-1


	def log(self, x):
		if x == 0:
			return 0
		else:
			return math.log(x,2)

class Node:
	def __init__(self,isLeaf, label, threshold):
		self.label = label
		self.threshold = threshold
		self.isLeaf = isLeaf
		self.children = []


