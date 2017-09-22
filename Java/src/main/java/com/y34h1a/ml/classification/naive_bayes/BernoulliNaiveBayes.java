package com.y34h1a.ml.classification.naive_bayes;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.ParseException;

import smile.classification.NaiveBayes;
import smile.feature.Bag;
import smile.math.Math;
import smile.validation.CrossValidation;

/**
 * Created by y34h1a on 9/22/17.
 */

public class BernoulliNaiveBayes {
    public static void main(String args[]) throws IOException, ParseException {
        String fileName = "/home/y34h1a/StudioProjects/Mechine Learning Examples/Data/movie.txt";

        String[] feature = {
                "outstanding", "wonderfully", "wasted", "lame", "awful", "poorly",
                "ridiculous", "waste", "worst", "bland", "unfunny", "stupid", "dull",
                "fantastic", "laughable", "mess", "pointless", "terrific", "memorable",
                "superb", "boring", "badly", "subtle", "terrible", "excellent",
                "perfectly", "masterpiece", "realistic", "flaws"
        };

        double[][] moviex;
        int[] moviey;

        String[][] x = new String[2000][];
        int[] y = new int[2000];

        try(BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)))) {
            for (int i = 0; i < x.length; i++) {
                String[] words = input.readLine().trim().split(" ");

                if (words[0].equalsIgnoreCase("pos")) {
                    //System.out.println("Words: " + words[0]);
                    y[i] = 1;
                } else if (words[0].equalsIgnoreCase("neg")) {
                    //System.out.println("Words: " + words[0]);
                    y[i] = 0;
                } else {
                    System.out.println("Invalid class label: " + words[0]);
                }

                x[i] = words;
            }
        } catch (IOException ex) {
            System.err.println(ex);
        }
        moviex = new double[x.length][];
        moviey = new int[y.length];
        Bag<String> bag = new Bag<>(feature);
        for (int i = 0; i < x.length; i++) {
            moviex[i] = bag.feature(x[i]);
            moviey[i] = y[i];
        }

        int n = x.length;
        int k = 10;
        CrossValidation cv = new CrossValidation(n, k);
        int error = 0;
        int total = 0;
        int errorlabel = 0;


        for (int i = 0; i < k; i++) {
            double[][] trainx = Math.slice(moviex, cv.train[i]);
            int[] trainy = Math.slice(moviey, cv.train[i]);
            NaiveBayes bayes = new NaiveBayes(NaiveBayes.Model.BERNOULLI, 2, feature.length);
            bayes.learn(trainx, trainy);

            double[][] testx = Math.slice(moviex, cv.test[i]);
            int[] testy = Math.slice(moviey, cv.test[i]);

            for (int j = 0; j < testx.length; j++) {
                int label = bayes.predict(testx[j]);
                if (label != -1) {
                    total++;
                    if (testy[j] != label) {
                        error++;
                    }
                }else {
                    errorlabel++;
                }
            }
        }

        System.out.format("Bernoulli error = %d of %d%n", error, total);

    }
}
