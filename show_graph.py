import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print('Provide the file name')
    exit(1)

# Load and parse the XML file
tree = ET.parse(sys.argv[1])
root = tree.getroot()

# Extract vertices
vertices = { }
for vertex in root.find('vertices'):
    idx = vertex.get('v')
    x = float(vertex.find('x').text)
    y = float(vertex.find('y').text)
    vertices[idx] = (x, y)

# Extract edges
edges = [ ]
for edge in root.find('edges'):
    tail = edge.find('tail').text
    head = edge.find('head').text
    edges.append((tail, head))

# Plotting
fig, ax = plt.subplots()
# Draw edges as arrows
for tail, head in edges:
    x1, y1 = vertices[tail]
    x2, y2 = vertices[head]
    ax.annotate("",
                xy=(x2, y2), xycoords='data',
                xytext=(x1, y1), textcoords='data',
                arrowprops=dict(arrowstyle="->", lw=2))
# Draw nodes and labels
for idx, (x, y) in vertices.items():
    ax.plot(x, y, 'o', color='blue')
    ax.text(x, y, idx, fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle,pad=0.3'))
ax.set_aspect('equal')
ax.axis('off')
plt.show()
