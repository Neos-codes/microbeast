import csv


name = input("Nombre archivo .csv: ")


count = 0
episode = 0
rewards = []
steps = []

with open(name+".csv", "r") as data:
    with open(name+"_processed.csv", "w") as new_data:
        csv_r = csv.reader(data, delimiter=",")
        csv_w = csv.writer(new_data)
        first_row_flag = True

        for row in csv_r:
            if first_row_flag:
                csv_w.writerow(row)
                first_row_flag = False

            else:
                count += 1
                rewards.append(float(row[0]))
                steps.append(float(row[1]))

                if count == 10:
                    csv_w.writerow([episode, sum(rewards)/len(rewards),
                                    sum(steps)/len(rewards)])
                    rewards.clear()
                    steps.clear()
                    count = 0
                    episode += 1

        # Si la ultima linea no alcanza a contar 10...
        if count != 0:
            csv_w.writerow([sum(rewards)/len(rewards),
                            sum(steps)/len(rewards)])
            rewards.clear()
            steps.clear()
            count = 0





