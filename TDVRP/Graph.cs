using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TDVRP;

class Vertex
{
    double Order;
    double ServiceTime;
    List<Edge> Edges;

    Vertex(double order, double serviceTime)
    {
        Order = order;
        ServiceTime = serviceTime;
        Edges = new List<Edge>();
    }

    Vertex(): this(0, 0) { }
}

class Edge
{
    Vertex End;
    double Distance;
    //double Speed(uint instant);
}

class Graph
{
    List<Vertex> vertices;

    Graph()
    {
        vertices = new List<Vertex>();
    }

    /* Manages construction and yields a graph whose layout cannot be modified from outside 
       of the class. */
    public class Builder
    {

    }
}
