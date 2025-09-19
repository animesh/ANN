using System;
using System.Diagnostics;
namespace ANN4CS
{
    class Program
    {
        static void Main(string[] args)
        {
            double[] input = new double[] { 0.05, 0.10 };
            double[,] inpw = new double[,] { { 0.15, 0.20 }, { 0.25, 0.3 } };
            double[] hidden = new double[2];
            double[,] hidw = new double[,] { { 0.4, 0.45 }, { 0.5, 0.55 } };
            double[] outputc = new double[2];
            double[] outputr = new double[] { 0.01, 0.99 };
            double[] bias = new double[] { 0.35, 0.6 };
            double[] cons = new double[] { 1, 1 };
            double lr = 0.5;
            double error = 1;
            double iter = 0;

            while (iter < 10000)
            {
              Console.WriteLine("Iter {0}<=>Error {1}",iter,error);
                iter++;
                error = 0;

                for (int j = 0; j < inpw.GetLength(0); j++)
                {
                    double collin = 0;
                    for (int i = 0; i < input.Length; i++)
                    {
                        collin += inpw[j, i] * input[i];
                    }
                    collin += bias[0] * cons[0];
                    collin = 1 / (1 + Math.Pow(Math.E, -1 * collin));
                    Console.WriteLine("collin {0}<=> hidden {1}",collin,hidden[j]);
                    hidden[j] = collin;

                }
                for (int j = 0; j < hidw.GetLength(0); j++)
                {
                    double collin = 0;
                    for (int i = 0; i < hidden.Length; i++)
                    {
                        collin += hidw[j, i] * hidden[i];
                    }
                    collin += bias[1] * cons[1];
                    collin = 1 / (1 + Math.Pow(Math.E, -collin));
                    Console.WriteLine("collin {0}<=> outputc {1}",collin,outputc[j]);
                    outputc[j] = collin;
                    error += Math.Pow(outputr[j] - outputc[j], 2) / 2;
                }

                for (int i = 0; i < input.Length; i++)
                {
                    for (int j = 0; j < inpw.GetLength(0); j++)
                    {
                        double delin = 0;
                        for (int k = 0; k < hidw.GetLength(0); k++)
                        {
                            delin += (outputc[k] - outputr[k])*outputc[k] * (1 - outputc[k]) * hidw[k, j];
                            //Console.WriteLine("delin{0},outputc{1},outputr{2},hidw{3}",delin,outputc[k],outputr[k],hidw[k,j]);
                        }
                        inpw[j, i] -= lr*delin*hidden[j] * (1 - hidden[j]) * input[i];
                        Console.WriteLine("inpw{0},delin{1},hidden{2},input{3},diff{4}",inpw[j, i] ,delin,hidden[j],input[i],lr*delin*hidden[j] * (1 - hidden[j]) * input[i]);
                    }
                }

                for (int i = 0; i < hidden.Length; i++)
                {
                    for (int j = 0; j < hidw.GetLength(0); j++)
                    {
                      Console.WriteLine("{0}-{1}*({2}-{3})*{2}*(1-{2})*{4}=>{5}",hidw[j,i],lr,outputc[j],outputr[j],hidden[i],lr*(outputc[j]-outputr[j])*(1-outputc[j])*outputc[j]*hidden[i]);
                        hidw[j, i] -= lr * (outputc[j] - outputr[j]) * outputc[j] * (1 - outputc[j]) * hidden[i];
			                     Console.WriteLine("j{0},i{1},hidw=>{2}",j,i,hidw[j,i]);
                    }
                }
                Console.WriteLine("Iteration = {0}\tError = {1}\tOutputs = {2}\t{03}", iter, error, outputc[0], outputc[1]);
                Debug.Write(iter + Environment.NewLine + error + Environment.NewLine + outputc[0] + Environment.NewLine + outputc[1] + Environment.NewLine);


            }
        }
    }
}
