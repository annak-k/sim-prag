import matplotlib.pyplot as plt
import numpy as np

labels = ["lit filled", "lit single-signal", "lit ambiguous", "lit max informative", "prag filled", "prag single-signal", "prag ambiguous", "prag max informative"]
colors = ['darkseagreen', 'steelblue', 'mediumpurple', 'darkorange']
bn60 = [0.0764110391314391, 0.0, 0.058884990248446615, 0.05639818204501164, 0.07562589637500482, 0.07629279036279307, 0.063332338180910525, 0.0540659242546408]
prior = [1/18 for _ in range(len(bn60))]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(5,5))
rects2 = ax.bar(x - width/2, prior, width, label='Prior', color=colors[1])
rects1 = ax.bar(x + width/2, bn60, width, label='Stationary distribution', color=colors[3])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('proportion of distribution')
plt.ylim(0,0.1)
plt.xticks(x, labels, rotation=45, ha="right", rotation_mode="anchor")
ax.legend()

fig.tight_layout()

plt.savefig('plots/combined_plot.png')
plt.close()

labels = ["filled", "single-signal", "ambiguous", "max informative"]
bn60 = [0.15203693550644392, 0.07629279036279307, 0.12221732842935714, 0.11046410629965244]
prior = [1/9 for _ in range(len(bn60))]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(5,5))
rects2 = ax.bar(x - width/2, prior, width, label='Prior', color=colors[1])
rects1 = ax.bar(x + width/2, bn60, width, label='Stationary distribution', color=colors[3])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('proportion of distribution')
plt.ylim(0,0.2)
plt.yticks(np.arange(0, 0.21, step=0.05))
plt.xticks(x, labels, rotation=45, ha="right", rotation_mode="anchor")
ax.legend()

fig.tight_layout()

plt.savefig('plots/lex_combined_plot.png')
plt.close()

labels = ["literal", "pragmatic"]
bn60 = [0.4247473642152489, 0.575252635784751]
prior = [1/2 for _ in range(len(bn60))]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(5,5))
rects2 = ax.bar(x - width/2, prior, width, label='Prior', color=colors[1])
rects1 = ax.bar(x + width/2, bn60, width, label='Stationary distribution', color=colors[3])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('proportion of distribution')
plt.ylim(0,1)
plt.xticks(x, labels, rotation=45, ha="right", rotation_mode="anchor")
ax.legend()

fig.tight_layout()

plt.savefig('plots/com_combined_plot.png')
plt.close()

states = ["lit filled", "lit single-signal", "lit ambiguous", "lit max informative", "prag filled", "prag single-signal", "prag ambiguous", "prag max informative"]

harvest = np.array([[0.15, 0., 0.04, 0.05, 0.16, 0.34, 0.23, 0.06],
 [0.00, 0.995, 0.005, 0.00, 0.00, 0.00, 0.00, 0.00],
 [0.01, 0.00, 0.85, 0.02, 0.02, 0.01, 0.09, 0.01],
 [0.03, 0.00, 0.02, 0.29, 0.02, 0.05, 0.36, 0.24],
 [0.21, 0.00, 0.03, 0.04, 0.13, 0.29, 0.27, 0.05],
 [0.15, 0.00, 0.05, 0.06, 0.17, 0.36, 0.20, 0.03],
 [0.08, 0.00, 0.08, 0.14, 0.08, 0.15, 0.33, 0.15],
 [0.04, 0.00, 0.03, 0.27, 0.04, 0.06, 0.28, 0.29]])


fig, ax = plt.subplots()
im = ax.imshow(harvest, cmap="inferno", vmin=0, vmax=1)
# ax.pcolorfast(harvest, cmap="inferno", vmin=0, vmax=1)
plt.colorbar(im)
# We want to show all ticks...
ax.set_xticks(np.arange(len(states)))
ax.set_yticks(np.arange(len(states)))
# ... and label them with the respective list entries
ax.set_xticklabels(states)
ax.set_yticklabels(states)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(states)):
    for j in range(len(states)):
        if (i == 1 and j == 1) or (i == 2 and j == 2):
            text = ax.text(j, i, harvest[i, j],
            ha="center", va="center", color="black")
        else:
            text = ax.text(j, i, harvest[i, j],
                 ha="center", va="center", color="w")

ax.set_title("Transition matrix for bottleneck 60")
fig.tight_layout()
plt.savefig('plots/heatmap_plot.png')