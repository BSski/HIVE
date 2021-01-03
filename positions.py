# Generate three 43x43 empty grid lists.
carnivores_pos = []


for i in range(0, 43):
    carnivores_pos.append([])
    for j in range(0, 43):
        carnivores_pos[i].append([])





carnivores_pos_hex = []
boundary_tiles = []
diameter = 45
for j in range((int(diameter/2)+1),(diameter*2)-1):
    row = []
    if j > diameter:
        if diameter - (j-diameter) < int(diameter/2)+1:
            break
        else:
            row_quantity = diameter - (j-diameter)
            boundary_tiles.append([row_quantity-1, j-(int(diameter/2)+1)])
    else:
        row_quantity = j
        boundary_tiles.append([row_quantity-1, j-(int(diameter/2)+1)])
    for i in range(0, row_quantity):
        row.append([])

    carnivores_pos_hex.append(row)

#print(carnivores_pos_hex[0])

for i in range(len(carnivores_pos_hex[0])):
    boundary_tiles.append([i, 0])

for i in range(len(carnivores_pos_hex[diameter-1])):
    boundary_tiles.append([i, diameter-1])

for i in range(diameter-1):
    boundary_tiles.append([0, i])



# print("\n", boundary_tiles)
