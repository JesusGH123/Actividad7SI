# References: None

#read the files
jobs = open ("jobs_training.txt", "r") 
job_validation = open ("jobs_validation.txt", "r") 

#We save all the jobs already classified into dictionary calle jobs_dict
jobs_dict = {}
for job in jobs:
  job = job.rstrip('\n')
  #We get the keys of the dictionary based on the departments that are un uppercase
  if job.isupper():
    key = job
    jobs_dict[key] = []
    continue
  jobs_dict[key].append(job)

#Then we save the jobs to classify in a dictionary called job_word_validation
job_word_validation = []
for line in job_validation:
  stripped_line = line.strip()
  line_list = stripped_line.split()
  job_word_validation.append(line_list)

department_frequency = {}
number_of_departments = 0

#Here we obtain the frequency of how many jobs belong to each department. And then we obtain the total number of jobs. We did that to calculate p(Ck) later
for jobs in jobs_dict:
  department_frequency[jobs] = []
  department_frequency[jobs].append(len(jobs_dict[jobs]))
  number_of_departments += len(jobs_dict[jobs])
      
#Get every word of jobs_training data set. The result will be a list of lists
job_words_splitted = []
for line in jobs_dict.values():
  for word in line:
    stripped_line = word.strip()
    line_list = stripped_line.split()
    job_words_splitted.append(line_list)
    
#Then, we flat the list so we will have only a list with all the differents words of the jobs to classify
flat_list = [item for sublist in job_words_splitted for item in sublist]

#We save the words in the list called jobs_flatten
jobs_flatten = []
for department in flat_list:
  if department not in jobs_flatten:
    jobs_flatten.append(department)


#This function will calculate all the probabilities of every department p(Xi|Ck)
def naive_bayes(words_of_job_to_validate):
  probabilities_by_department = {}
  for department,jobs in jobs_dict.items():
    probabilities_by_department[department] = []
    jobs_by_department = department_frequency[department][0]
    occurrences = 0
    probability = 1
    
    for job in jobs_flatten:
      #If the word is part of the job I looking for then i will count the occurence of appearing
      if job in words_of_job_to_validate:
        for job_department in jobs:
          if job in job_department:
            occurrences = occurrences + 1
        #Calculate the probability for each word
        probability = probability * (occurrences/jobs_by_department)
        occurrences = 0
      if job not in words_of_job_to_validate:
      #If the word is not part of the job I want to classify then i will count the occurence of not appearing
        for job_department in jobs:
          if job not in job_department:
            occurrences = occurrences + 1
        probability = probability * (occurrences/jobs_by_department)
        occurrences = 0
    probabilities_by_department[department].append(probability)

  #These are all the probabilities by department
  print("First probabilities: ",probabilities_by_department,"\n")
  total_probability =0

  #Here we calculate p(x)
  for x in jobs_dict.keys():
    total_probability += probabilities_by_department[x][0] * department_frequency[x][0] / number_of_departments
  
  print("Total probability: ",total_probability,"\n")

  #Here we calculate p(Ck|Xi)
  final_probabilities_by_department = {}
  for key in jobs_dict.keys():
    final_probabilities_by_department[key] = []
    final_probabilities_by_department[key].append(probabilities_by_department[key][0]*(department_frequency[key][0] / number_of_departments) / total_probability)

  #These are the final probabilities and we obtain the maximum in order to know in which departmen the job will be classified
  print("Final probabilities: ", final_probabilities_by_department,"\n")
  return (max(final_probabilities_by_department, key=final_probabilities_by_department.get))

#Here we classify every job in jobs_validation.txt
for job in job_word_validation:
  #Here we unify each word of a job in a unique string
  print("Job: ", ' '.join(job))
  print("The job: ", ' '.join(job), " was classified in ",naive_bayes(job)," department\n\n")
  