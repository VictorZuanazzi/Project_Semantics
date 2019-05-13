import sys

def create_dataset_list(suffix):
	data_list = list()
	with open("s1." + suffix, "r") as f1:
		with open("s2." + suffix, "r") as f2:
			with open("labels." + suffix, "r") as f_lab:
				for line in zip(f1.readlines(), f2.readlines(), f_lab.readlines()):
					data_list.append(line)
	return data_list


test_set = create_dataset_list("test")
hard_test_set = create_dataset_list("test_hard")
easy_test_set = list()
hard_test_set_index = 0

for test_set_index in range(len(test_set)):
	if hard_test_set_index < len(hard_test_set) and all([test_set[test_set_index][i] == hard_test_set[hard_test_set_index][i] for i in range(3)]):
		hard_test_set_index += 1
	else:
		easy_test_set.append(test_set[test_set_index])

if len(test_set) != len(easy_test_set) + len(hard_test_set):
	print("List lengths are not correct...")
	sys.exit(1)

s1 = ""
s2 = ""
labels = ""
for easy_data in easy_test_set:
	sent_1, sent_2, lab = easy_data
	s1 += sent_1
	s2 += sent_2 
	labels += lab

with open("s1.test_easy", "w") as f:
	f.write(s1)

with open("s2.test_easy", "w") as f:
	f.write(s2)

with open("labels.test_easy", "w") as f:
	f.write(labels)