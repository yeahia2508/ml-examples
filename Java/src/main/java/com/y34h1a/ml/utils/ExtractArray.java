package com.y34h1a.ml.utils;

/**
 * Created by y34h1a on 9/13/17.
 */

public class ExtractArray {

    public static  double[][] deleteColumn(double[][] args,int col)
    {
        double[][] nargs = new double[0][];
        if(args != null && args.length > 0 && args[0].length > col)
        {
            nargs = new double[args.length][args[0].length-1];
            for(int i=0; i<args.length; i++)
            {
                int newColIdx = 0;
                for(int j=0; j<args[i].length; j++)
                {
                    if(j != col)
                    {
                        nargs[i][newColIdx] = args[i][j];
                        newColIdx++;
                    }
                }
            }
        }
        return nargs;
    }

    public static  double[] convertSingleColumn(double[][] args,int selectedColumn)
    {
        double[] nargs = new double[0];
        if(args != null && args.length > 0)
        {
            nargs = new double[args.length];

            for(int i=0; i<args.length; i++)
            {
                for(int j=0; j < args[i].length; j++)
                {
                    if(j == selectedColumn)
                    {
                        nargs[i] = args[i][j];
                    }
                }
            }
        }
        return nargs;
    }

    public static double[][] deleteColumns(double[][] args,int colIndexStart, int colIndexEnd)
    {
        double[][] nargs = new double[0][];
        if(args != null && args.length > 0 && args[0].length > colIndexStart && args[0].length > colIndexEnd && (colIndexEnd > colIndexStart))
        {

            int diff = colIndexEnd - colIndexStart;
            nargs = new double[args.length][args[0].length - (diff + 1)];

            for(int i=0; i< args.length; i++)
            {
                int newColIdx = 0;
                for(int j=0; j< args[i].length; j++)
                {
                    if (j < colIndexStart || j > colIndexEnd){

                        nargs[i][newColIdx] = args[i][j];
                        newColIdx++;
                    }
                }
            }
        }

        return nargs;
    }

    public static double[][] sliceIndex(double[][] args, int[] colIndexs)
    {
        double[][] nargs = new double[0][];
        if(args != null && colIndexs.length > 0 && args[0].length >= colIndexs.length)
        {
            nargs = new double[args.length][args[0].length - (colIndexs.length)];

            for(int i=0; i< args.length; i++)
            {
                int newColIdx = 0;
                for(int j=0; j< args[i].length; j++)
                {
                    boolean found = false;

                    for (int colIndex : colIndexs) {
                        if (colIndex  == j) {
                            found = true;
                            break;
                        }
                    }

                    if (!found){
                        nargs[i][newColIdx] = args[i][j];
                        newColIdx++;
                    }
                }
            }
        }

        return nargs;
    }


}
