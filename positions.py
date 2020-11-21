# Generate three 43x43 empty grid lists.
herbs_pos = []
herbivores_pos = []
carnivores_pos = []
for i in range(0, 43):
    herbs_pos.append([])
    for j in range(0, 43):
        herbs_pos[i].append([])

for i in range(0, 43):
    herbivores_pos.append([])
    for j in range(0, 43):
        herbivores_pos[i].append([])

for i in range(0, 43):
    carnivores_pos.append([])
    for j in range(0, 43):
        carnivores_pos[i].append([])
