package nytuaai

import scala.io.Source
import java.io._
import scala.util.Random
import scala.math.pow
import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ArrayBuffer

object DeBruijn {
  
  def main(args: Array[String]) {
  
    System.setProperty("hadoop.home.dir", "E:/hadoop");
    
    val conf = new SparkConf().setAppName("DeBruijn").setMaster("local")
    val sc = new SparkContext(conf)
  
    //HYPERPARAMETER: length of kmers
    val k = 3
    //HYPERPARAMTERS: probability threshold for dropping vertices/edges from the graph
    val edge_drop_filter = 0.0
    
    /*
    //PREPROCESS FASTQ FILE BEFORE PUTTING IT ONTO HDFS
    val file = new File("E:/downloads/dna/frag_1_only_reads_and_quality")
    val bw = new BufferedWriter(new FileWriter(file))
    var i = 0
    for (line <- Source.fromFile("E:/downloads/dna/frag_1.fastq").getLines) {
      if (i%4==1) {
        bw.write(line)
      }
      else if (i%4==3) {
        bw.write(line+"\n")
      }
      i += 1
    }
    bw.close()
    */
    
    //val filename = "E:dna/frag_1_only_reads_and_quality"
    val filename = "E:dna/short_short_short_frag_only_reads_and_quality"
    //val lines = sc.textFile(filename)
    val lines = sc.parallelize(Array("AATCT~~~~~","TGTAA~~~~~","GATTA~~~~~","ACAGA~~~~~"))
    
    def quality_code_to_percentage(char:Char) = {
      (1 - pow(10,-(char.toInt-33)/10.0) + 0.25*pow(10,-(char.toInt-33)/10.0) )
    }
    
    val dnas_with_qualities_coded = lines
    .map(line => (line.slice(0,line.length/2), line.slice(line.length/2,line.length).toArray.map(quality_code => quality_code_to_percentage(quality_code))))
    
    def dna_to_rep(dna:String) = {
      val coded_dna = dna.replace('A', '0').replace('C', '1').replace('G', '2').replace('T', '3').replace('N', '4')
      java.lang.Long.parseLong(coded_dna, 5)
      //what about reverse complement???
    }
    
    def rep_to_dna(rep:Long) = {
      var dna = java.lang.Long.toString(rep, 5)
      while (dna.length != k-1) {
        dna = "0" + dna
      }
      dna.replace('0', 'A').replace('1', 'C').replace('2', 'G').replace('3', 'T').replace('4', 'N')
      //what about reverse complement???
    }

    //string of vertices that form the edge, quality of edge
    val list_of_edges = dnas_with_qualities_coded.map(dna_with_quality_coded =>
      for {i <- 0 until dna_with_quality_coded._1.length - k + 1
        quality = dna_with_quality_coded._2.slice(i,i+k).reduce(_*_)
        if quality > edge_drop_filter
      }
        yield (dna_with_quality_coded._1.slice(i,i+k),
               dna_with_quality_coded._2.slice(i,i+k).reduce(_*_)))
        
    //(ID of vertex, indegree of vertex, outdegree of vertex ID of next vertex (only useful when vertex has outdegree=1))
    val list_of_kmers = list_of_edges.map(edges =>
      for {i <- 0 until edges.length
        j <- 0 until 2
        if (edges.length > 0)
          if (i==0 || j==1 || edges(i)._1.dropRight(1) != edges(i-1)._1.drop(1))
      }
        yield (if (j==0) dna_to_rep(edges(i)._1.dropRight(1)) else dna_to_rep(edges(i)._1.drop(1)),
               (if (i == 0 && j==0) 0 else if (j==0 && edges(i-1)._1.drop(1) != edges(i)._1.dropRight(1)) 0 else 1,
                if (i == edges.length - 1 && j==1) 0 else if (j==1 && edges(i)._1.drop(1) != edges(i+1)._1.dropRight(1)) 0 else 1,
                if (i == edges.length - 1 && j==1) -1L else if (j==1 && edges(i)._1.drop(1) != edges(i+1)._1.dropRight(1)) -1L  else if (j==0) dna_to_rep(edges(i)._1.drop(1)) else dna_to_rep(edges(i+1)._1.drop(1)))))
    
    
    val reduced_kmers = list_of_kmers
    .flatMap(kmers => kmers)
    .reduceByKey((a,b) => (a._1+b._1, a._2+b._2, a._3))
    .map(kmer => (kmer._1, (rep_to_dna(kmer._1), kmer._2._1, kmer._2._2, kmer._2._3, scala.util.Random.nextBoolean())))
    
    val edges_with_weights = list_of_edges
    .flatMap(edges => edges)
    .reduceByKey((a,b) => a+b)
    .map(edge => (dna_to_rep(edge._1.dropRight(1)), dna_to_rep(edge._1.drop(1)), edge._2))
    
    val edges = edges_with_weights.map(edge => Edge(edge._1, edge._2, edge._3))
    /*val edges = edges_with_weights
    .map(edge => List.fill(edge._3.toInt)(Edge(edge._1, edge._2, 1.0)))
    .flatMap(edges => edges)*/
    
    var deBruijn = Graph(reduced_kmers, edges)

    println(deBruijn.numEdges)
    println(deBruijn.numVertices)
    deBruijn.vertices.collect().foreach(println)
    deBruijn.edges.collect().foreach(println)
    //deBruijn.vertices.saveAsTextFile("vertices")
    //deBruijn.edges.saveAsTextFile("edges")

    //--BEGIN---VERTEX MERGING---
    
    val initialMsg1 = ("initialMsg", -1, -1, -1L)
    
    def vprog1(vertexId: VertexId, value: (String,Int,Int,Long,Boolean), message: (String,Int,Int,Long)): (String,Int,Int,Long,Boolean) = {
      if (message == initialMsg1)
        value
      else
        if (message == ("nothingToDoHere", -1, -1, -1L))
          (value._1, value._2, value._3, value._4, false)
        else
          if (message == ("potentialExists", -1, -1, -1L))
            (value._1, value._2, value._3, value._4, true)
          else
            (message._1, message._2, message._3, message._4, true)
    }
    
    def sendMsg1(triplet: EdgeTriplet[(String,Int,Int,Long,Boolean), Double]): Iterator[(VertexId, (String,Int,Int,Long))] = {
    
      if (triplet.srcAttr._3 <= 1 && triplet.dstAttr._2 <= 1 && triplet.dstAttr._3 <= 1)
        if  (triplet.srcAttr._5 == true && triplet.dstAttr._5 == false)
          Iterator((triplet.srcId, (triplet.srcAttr._1+triplet.dstAttr._1.takeRight(triplet.dstAttr._1.length-(k-2))+" "+triplet.dstId.toString, triplet.srcAttr._2, triplet.dstAttr._3, triplet.dstAttr._4)))
        else
          Iterator((triplet.srcId, ("potentialExists", -1, -1, -1L)))
      else
        Iterator((triplet.srcId, ("nothingToDoHere", -1, -1, -1L)))
    }
    
    def mergeMsg1(msg1: (String,Int,Int,Long), msg2: (String,Int,Int,Long)): (String,Int,Int,Long) = msg1
    
    do {
      deBruijn = Graph(deBruijn.vertices.map(vertex => (vertex._1, (vertex._2._1, vertex._2._2, vertex._2._3, vertex._2._4, scala.util.Random.nextBoolean()))), deBruijn.edges)
      
      deBruijn = deBruijn.pregel(initialMsg1, 
                              1, 
                              EdgeDirection.Out)(
                              vprog1,
                              sendMsg1,
                              mergeMsg1)
      
      var verticesToDelete = sc.parallelize(Array[(VertexId, (String,Int,Int,Long,Boolean))]())
      var edgesToDelete = sc.parallelize(Array[Edge[Double]]())
      var edgesToAdd = sc.parallelize(Array[Edge[Double]]())
      
      var filteredVertices = deBruijn.vertices.filter(vertex => vertex._2._1.split(" ").length == 2).collect()
      
      for (filteredVertex <- filteredVertices) {
        verticesToDelete = verticesToDelete.union(deBruijn.vertices.filter(vertex => vertex._1 == filteredVertex._2._1.split(" ")(1).toLong))
        edgesToDelete = edgesToDelete.union(deBruijn.edges.filter(edge => edge.srcId == filteredVertex._2._1.split(" ")(1).toLong || edge.dstId == filteredVertex._2._1.split(" ")(1).toLong))
        if (filteredVertex._2._4 >= 0) {
          edgesToAdd = edgesToAdd.union(sc.parallelize(Array(Edge(filteredVertex._1, filteredVertex._2._4,
              deBruijn.edges.filter(edge => edge.srcId == filteredVertex._1).first().attr
              *
              deBruijn.edges.filter(edge => edge.dstId == filteredVertex._2._4).first().attr))))
        }
      }
      
      deBruijn = Graph(deBruijn.vertices.subtract(verticesToDelete).map(vertex => (vertex._1, (vertex._2._1.split(" ")(0), vertex._2._2, vertex._2._3, vertex._2._4, vertex._2._5))), deBruijn.edges.union(edgesToAdd).subtract(edgesToDelete))
    
    } while (deBruijn.vertices.filter(vertex => vertex._2._5 == true).count() > 0)
      
    deBruijn.vertices.collect().foreach(println)
    deBruijn.edges.collect().foreach(println)
    //deBruijn.vertices.saveAsTextFile("mergedVertices")
    //deBruijn.edges.saveAsTextFile("mergedEdges")
    
    //--END---VERTEX MERGING---
    
    //--BEGIN---TIP REMOVAL---
    val initialMsg2 = "initialMsg"
    
    def vprog2(vertexId: VertexId, value: (String,Int,Int,Long,Boolean), message: String): (String,Int,Int,Long,Boolean) = {
      if (message == initialMsg2)
        value
      else
        (message, value._2, value._3, value._4, value._5)
    }
    
    def sendMsg2(triplet: EdgeTriplet[(String,Int,Int,Long,Boolean), Double]): Iterator[(VertexId, String)] = {
    
      if (triplet.dstAttr._3 == 0 && triplet.dstAttr._1.length <= 2*k)
        Iterator((triplet.dstId, "KILL ME!"))
      else
        Iterator.empty
    }
    
    def mergeMsg2(msg1: String, msg2: String): String = msg1
    
    deBruijn = deBruijn.pregel(initialMsg2, 
                            1, 
                            EdgeDirection.Out)(
                            vprog2,
                            sendMsg2,
                            mergeMsg2)
    
    var verticesToUpdate = sc.parallelize(Array[(VertexId, (String,Int,Int,Long,Boolean))]())
    var edgesToDelete = sc.parallelize(Array[Edge[Double]]())
    var edgesToAdd = sc.parallelize(Array[Edge[Double]]())
    var updatedVertices = deBruijn.vertices
    var verticesToDelete = deBruijn.vertices.filter(vertex => vertex._2._1 == "KILL ME!")
    var vertexToUpdate = updatedVertices.first()
    
    var filteredVertices = verticesToDelete.collect()
    for (filteredVertex <- filteredVertices) {
      var verticesToUpdate = deBruijn.edges.filter(edge => edge.dstId == filteredVertex._1).map(edge => edge.srcId).collect()
      //we have to update all the edges which pointed to our tip, and subtract 1 from their out-value
      for (v <- verticesToUpdate) {
        var vertex = updatedVertices.filter(vertex => vertex._1 == v).first()
        if (vertex._2._3 == 2)
          //if only one out-edge will be left after update, then we have to find that one and update the out-edge parameter with its ID)
          vertexToUpdate = (vertex._1, (vertex._2._1, vertex._2._2, 1, deBruijn.edges.filter(edge => edge.srcId == vertex._1 && edge.dstId != filteredVertex._1).first().dstId, vertex._2._5))
        else
          if (vertex._2._3 == 1)
            vertexToUpdate = (vertex._1, (vertex._2._1, vertex._2._2, 0, -1L, vertex._2._5))
          else
            vertexToUpdate = (vertex._1, (vertex._2._1, vertex._2._2, vertex._2._3 - 1, vertex._2._4, vertex._2._5))
        updatedVertices = VertexRDD(updatedVertices.map(v => if (v._1 != vertex._1) v else vertexToUpdate))
      }
      edgesToDelete = edgesToDelete.union(deBruijn.edges.filter(edge => edge.dstId == filteredVertex._1))
    }
    
    deBruijn = Graph(updatedVertices.subtract(verticesToDelete), deBruijn.edges.union(edgesToAdd).subtract(edgesToDelete))
    
    deBruijn.vertices.collect().foreach(println)
    deBruijn.edges.collect().foreach(println)
    //deBruijn.vertices.saveAsTextFile("tipRemovedVertices")
    //deBruijn.edges.saveAsTextFile("tipRemovedEdges")
    
    //--END---TIP REMOVAL---
    //--BEGIN---VERTEX MERGING---
    do {
      deBruijn = Graph(deBruijn.vertices.map(vertex => (vertex._1, (vertex._2._1, vertex._2._2, vertex._2._3, vertex._2._4, scala.util.Random.nextBoolean()))), deBruijn.edges)
      
      deBruijn = deBruijn.pregel(initialMsg1, 
                              1, 
                              EdgeDirection.Out)(
                              vprog1,
                              sendMsg1,
                              mergeMsg1)
      
      var verticesToDelete = sc.parallelize(Array[(VertexId, (String,Int,Int,Long,Boolean))]())
      var edgesToDelete = sc.parallelize(Array[Edge[Double]]())
      var edgesToAdd = sc.parallelize(Array[Edge[Double]]())
      
      var filteredVertices = deBruijn.vertices.filter(vertex => vertex._2._1.split(" ").length == 2).collect()
      
      for (filteredVertex <- filteredVertices) {
        verticesToDelete = verticesToDelete.union(deBruijn.vertices.filter(vertex => vertex._1 == filteredVertex._2._1.split(" ")(1).toLong))
        edgesToDelete = edgesToDelete.union(deBruijn.edges.filter(edge => edge.srcId == filteredVertex._2._1.split(" ")(1).toLong || edge.dstId == filteredVertex._2._1.split(" ")(1).toLong))
        if (filteredVertex._2._4 >= 0) {
          edgesToAdd = edgesToAdd.union(sc.parallelize(Array(Edge(filteredVertex._1, filteredVertex._2._4,
              deBruijn.edges.filter(edge => edge.srcId == filteredVertex._1).first().attr
              *
              deBruijn.edges.filter(edge => edge.dstId == filteredVertex._2._4).first().attr))))
        }
      }
      
      deBruijn = Graph(deBruijn.vertices.subtract(verticesToDelete).map(vertex => (vertex._1, (vertex._2._1.split(" ")(0), vertex._2._2, vertex._2._3, vertex._2._4, vertex._2._5))), deBruijn.edges.union(edgesToAdd).subtract(edgesToDelete))
    
    } while (deBruijn.vertices.filter(vertex => vertex._2._5 == true).count() > 0)
    
    deBruijn.vertices.collect().foreach(println)
    deBruijn.edges.collect().foreach(println)
    //deBruijn.vertices.saveAsTextFile("mergedVertices")
    //deBruijn.edges.saveAsTextFile("mergedEdges")
      
    //--END---VERTEX MERGING---
    
    println("PROGRAM STOPPED RUNNING")
  }
}