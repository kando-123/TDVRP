import json
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.widgets import Button

class GraphViewer:
    def __init__(self, graph):
        self.vertices = graph["vertices"]
        self.edges = graph["edges"]
        self.depot = graph.get("depot", "0")
        self.num_frames = len(self.edges[0]["w"]) if self.edges else 1
        self.current_frame = 0
        # Get min/max weights
        w_all = [w for edge in self.edges for w in edge["w"]]
        self.w_min = min(w_all)
        self.w_max = max(w_all)
        # Setup plot
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.22)
        self.axprev = plt.axes([0.29, 0.06, 0.16, 0.075])
        self.axnext = plt.axes([0.55, 0.06, 0.16, 0.075])
        self.bprev = Button(self.axprev, 'prev')
        self.bnext = Button(self.axnext, 'next')
        self.bprev.on_clicked(self.prev)
        self.bnext.on_clicked(self.next)
        self.draw_graph(self.current_frame)

    def draw_graph(self, idx):
        self.ax.clear()
        cmap = cm.RdYlGn_r
        norm = mcolors.Normalize(vmin=self.w_min, vmax=self.w_max)
        # Draw edges
        for edge in self.edges:
            u, v, w = edge["u"], edge["v"], edge["w"][idx]
            x1, y1 = self.vertices[u]["x"], self.vertices[u]["y"]
            x2, y2 = self.vertices[v]["x"], self.vertices[v]["y"]
            color = cmap(norm(w))
            self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, zorder=2)
        # Draw vertices
        for k, vdata in self.vertices.items():
            x, y = vdata["x"], vdata["y"]
            q = vdata.get("q", 0)
            if q:
                self.ax.text(x, y, f'{k}/{q}', ha='center', va='center', zorder=4, fontsize=7, color='red', weight='bold')
            elif k != self.depot:
                self.ax.text(x, y, k, ha='center', va='center', zorder=4, fontsize=6)
            else:
                self.ax.text(x, y, f'({k})', ha='center', va='center', zorder=4, fontsize=7, color='blue', weight='bold')

        self.ax.axis('equal')
        self.ax.axis('off')
        self.ax.set_title(f"Graph at time index {idx}", fontsize=16)
        self.fig.canvas.draw_idle()

    def next(self, event):
        if self.current_frame < self.num_frames - 1:
            self.current_frame += 1
            self.draw_graph(self.current_frame)

    def prev(self, event):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.draw_graph(self.current_frame)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py graph.json")
        return
    with open(sys.argv[1], "r") as f:
        graph = json.load(f)
    GraphViewer(graph)
    plt.show()

if __name__ == "__main__":
    main()
