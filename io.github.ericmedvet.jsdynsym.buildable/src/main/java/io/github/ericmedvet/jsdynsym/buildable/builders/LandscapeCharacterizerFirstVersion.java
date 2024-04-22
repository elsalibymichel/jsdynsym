/*-
 * ========================LICENSE_START=================================
 * jsdynsym-buildable
 * %%
 * Copyright (C) 2023 - 2024 Eric Medvet
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =========================LICENSE_END==================================
 */
package io.github.ericmedvet.jsdynsym.buildable.builders;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import io.github.ericmedvet.jnb.core.NamedBuilder;
import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction;
import io.github.ericmedvet.jsdynsym.control.Simulation;
import io.github.ericmedvet.jsdynsym.control.SingleAgentTask;
import io.github.ericmedvet.jsdynsym.control.navigation.NavigationEnvironment;
import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.MultiLayerPerceptron;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

public class LandscapeCharacterizerFirstVersion {

  private static final Logger L = Logger.getLogger(LandscapeCharacterizerFirstVersion.class.getName());

  record Pair(String environment, String builder) {}

  record Range(double min, double max) {}

  private static final List<String> FITNESS_FUNCTIONS = List.of("ds.e.n.avgD()", "ds.e.n.minD()", "ds.e.n.finalD()");
  private static final List<Pair> PROBLEMS = List.of(
      // BARRIER
      // Plot: x_axis=innerLayerRatio(1, 2, 3, 4, 5) with fixed nOfSensors=7 and Barrier=C_BARRIER
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 1)"),
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 2)"),
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 4)"),
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 5)"),

      // Plot: x_axis=nOfSensors(3, 5, 7, 9, 11) with fixed innerLayerRatio=3 and Barrier=C_BARRIER
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 3)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 5)", "ds.num.mlp(innerLayerRatio = 3)"),
      // new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 9)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 11)", "ds.num.mlp(innerLayerRatio = 3)"),

      // Plot: x_axis=barrier(A_BARRIER, B_BARRIER, C_BARRIER, D_BARRIER, E_BARRIER) with fixed innerLayerRatio=3
      // and nOfSensors=7
      new Pair("ds.e.navigation(arena = A_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = B_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      // new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = D_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = E_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),

      // MAZE
      // Plot: x_axis=innerLayerRatio(1, 2, 3, 4, 5) with fixed nOfSensors=7 and Barrier=C_MAZE
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 1)"),
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 2)"),
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 4)"),
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 5)"),

      // Plot: x_axis=nOfSensors(3, 5, 7, 9, 11) with fixed innerLayerRatio=3 and Barrier=C_MAZE
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 3)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 5)", "ds.num.mlp(innerLayerRatio = 3)"),
      // new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 9)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 11)", "ds.num.mlp(innerLayerRatio = 3)"),

      // Plot: x_axis=barrier(A_MAZE, B_MAZE, C_MAZE, D_MAZE, E_MAZE) with fixed innerLayerRatio=3 and
      // nOfSensors=7
      new Pair("ds.e.navigation(arena = A_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = B_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      // new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = D_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = E_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"));

  private static final NamedBuilder<Object> BUILDER = NamedBuilder.fromDiscovery();
  private static final String DEFAULT_FORMAT_PATH =
      "LandscapeCharacterizerFirstVersion__s=%d_np=%d_nn=%d_ns=%d_gb=[%.1f-%.1f]__%s.csv";

  public static class Configuration {

    @Parameter(
        names = {"--seed", "-s"},
        description = "Seed for the random number generator.")
    public long seed = 0;

    @Parameter(
        names = {"--nPoints", "-np"},
        description = "Number of points in the genotype space to consider.")
    public int nPoints = 5; // 50

    @Parameter(
        names = {"--nNeighbors", "-nn"},
        description = "Number of neighbors to consider for each point.")
    public int nNeighbors = 5; // 50

    @Parameter(
        names = {"--nSamples", "-ns"},
        description = "Number of samples for each couple of point and neighbor.")
    public int nSamples = 5; // 60

    @Parameter(
        names = {"--segmentLength", "-sl"},
        description = "Length of the segment.")
    public double segmentLength = 2;

    @Parameter(
        names = {"--genotypeBounds", "-gb"},
        description = "Bounds for the genotype components.")
    public List<Double> genotypeBoundsList = Arrays.asList(-3.0, 3.0);

    @Parameter(
        names = {"--resultsTarget", "-t"},
        description = "File path where to store the results.")
    public String resultsTarget = DEFAULT_FORMAT_PATH;

    @Parameter(
        names = {"--deltaUpdate", "-du"},
        description = "Delta [sec] update for the progress printer.")
    public int deltaUpdate = 10;

    @Parameter(
        names = {"--help", "-h"},
        description = "Show this help.",
        help = true)
    public boolean help;
  }

  public static Range getGenotypeBounds(List<Double> genotypeBoundsList) {
    if (genotypeBoundsList.size() != 2) {
      throw new IllegalArgumentException("GenotypeBounds requires exactly 2 arguments.");
    }
    return new Range(genotypeBoundsList.get(0), genotypeBoundsList.get(1));
  }

  @SuppressWarnings("unchecked")
  private static double[] getFitnessValues(Pair problem, double[] mlpWeights) {
    NavigationEnvironment environment = (NavigationEnvironment) BUILDER.build(problem.environment);
    MultiLayerPerceptron mlp = ((NumericalDynamicalSystems.Builder<MultiLayerPerceptron, ?>)
            NamedBuilder.fromDiscovery().build(problem.builder))
        .apply(environment.nOfOutputs(), environment.nOfInputs());
    SingleAgentTask<DynamicalSystem<double[], double[], ?>, double[], double[], NavigationEnvironment.State> task =
        SingleAgentTask.fromEnvironment(environment, new double[2], new DoubleRange(0, 60), 1 / 60d);
    mlp.setParams(mlpWeights);
    Simulation.Outcome<SingleAgentTask.Step<double[], double[], NavigationEnvironment.State>> outcome =
        task.simulate(mlp);
    return FITNESS_FUNCTIONS.stream()
        .mapToDouble(s -> ((FormattedNamedFunction<
                    Simulation.Outcome<
                        SingleAgentTask.Step<double[], double[], NavigationEnvironment.State>>,
                    Double>)
                BUILDER.build(s))
            .apply(outcome))
        .toArray();
  }

  @SuppressWarnings("unchecked")
  public static void main(String[] args) throws FileNotFoundException {

    Locale.setDefault(Locale.ROOT);

    Configuration configuration = new Configuration();
    JCommander jc = JCommander.newBuilder().addObject(configuration).build();
    jc.setProgramName(LandscapeCharacterizerFirstVersion.class.getName());
    try {
      jc.parse(args);
    } catch (ParameterException e) {
      e.usage();
      L.severe(String.format("Cannot read command line options: %s", e));
      System.exit(-1);
    } catch (RuntimeException e) {
      L.severe(e.getClass().getSimpleName() + ": " + e.getMessage());
      System.exit(-1);
    }

    // check help
    if (configuration.help) {
      jc.usage();
      System.exit(0);
    }

    final Range genotypeBounds = getGenotypeBounds(configuration.genotypeBoundsList);

    if (configuration.resultsTarget.equals(DEFAULT_FORMAT_PATH)) {
      ZonedDateTime timestamp = ZonedDateTime.now(); // Use ZonedDateTime
      DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH.mm.ss");
      configuration.resultsTarget = String.format(
          DEFAULT_FORMAT_PATH,
          configuration.seed,
          configuration.nPoints,
          configuration.nNeighbors,
          configuration.nSamples,
          genotypeBounds.min(),
          genotypeBounds.max(),
          timestamp.format(formatter));
    }

    try (CSVPrinter printer = new CSVPrinter(new FileWriter("csv.txt"), CSVFormat.EXCEL)) {
      printer.printRecord(
          "ENVIRONMENT",
          "BUILDER",
          "POINT_INDEX",
          "NEIGHBOR_INDEX",
          "SAMPLE_INDEX",
          "SEGMENT_LENGTH",
          "GENOTYPE_SIZE",
          "FITNESS_FUNCTIONS");
    } catch (IOException ex) {
      System.out.printf("Cannot create CSVPrinter: %s%n", ex);
      System.exit(0);
    }

    int totalSimulations =
        PROBLEMS.size() * configuration.nPoints * configuration.nNeighbors * configuration.nSamples;
    AtomicInteger counterSimulation = new AtomicInteger();
    long initialTime = System.currentTimeMillis();
    Runnable progressPrinterRunnable = () -> {
      System.out.printf("Simulations: %d/%d%n", counterSimulation.get(), totalSimulations);
      int totalMinutesRemaining = (int) Math.ceil((System.currentTimeMillis() - initialTime)
          / 1000.0
          / counterSimulation.get()
          * (totalSimulations - counterSimulation.get())
          / 60);
      int hours = totalMinutesRemaining / 60;
      int minutes = totalMinutesRemaining % 60;
      int days = hours / 24;
      if (days > 0) {
        System.out.printf("Remaining time estimate: %4d d  %2d h  %2d min%n", days, hours % 24, minutes);
      } else if (hours > 0) {
        System.out.printf("Remaining time estimate: %2d h  %2d min%n", hours, minutes);
      } else {
        System.out.printf("Remaining time estimate: %2d min%n", minutes);
      }
    };
    ScheduledExecutorService updatePrinterExecutor = Executors.newScheduledThreadPool(1);
    updatePrinterExecutor.scheduleAtFixedRate(
        progressPrinterRunnable, 0, configuration.deltaUpdate, TimeUnit.SECONDS);

    PrintStream ps = new PrintStream(configuration.resultsTarget);
    String header = "ENVIRONMENT,BUILDER,POINT_INDEX,NEIGHBOR_INDEX,SAMPLE_INDEX,SEGMENT_LENGTH,GENOTYPE_SIZE,"
        + String.join(",", FITNESS_FUNCTIONS);
    ps.println(header);
    ExecutorService executorService =
        Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() - 1);
    Random random = new Random(configuration.seed);

    for (Pair problem : PROBLEMS) {
      NavigationEnvironment environment = (NavigationEnvironment) BUILDER.build(problem.environment);
      MultiLayerPerceptron mlp = ((NumericalDynamicalSystems.Builder<MultiLayerPerceptron, ?>)
              NamedBuilder.fromDiscovery().build(problem.builder))
          .apply(environment.nOfOutputs(), environment.nOfInputs());
      int genotypeLength = mlp.getParams().length;

      for (int point = 0; point < configuration.nPoints; ++point) {
        double[] centralGenotype = IntStream.range(0, genotypeLength)
            .mapToDouble(i -> genotypeBounds.min()
                + random.nextDouble() * (genotypeBounds.max() - genotypeBounds.min()))
            .toArray();

        // Compute and store the centralGenotype fitness for the current point once and for all here
        int finalPoint = point;
        executorService.submit(() -> {
          double[] centralGenotypeFitnessValues = getFitnessValues(problem, centralGenotype);
          for (int n = 0; n < configuration.nNeighbors; ++n) {
            String line = "%s,%s,%d,%d,%d,%.2e,%d,"
                    .formatted(
                        problem.environment,
                        problem.builder,
                        finalPoint,
                        n,
                        0,
                        configuration.segmentLength,
                        genotypeLength)
                + Arrays.stream(centralGenotypeFitnessValues)
                    .mapToObj(value -> String.format("%.5e", value))
                    .collect(Collectors.joining(","));
            ps.println(line);
            counterSimulation.getAndIncrement();
          }
        });

        for (int neighbor = 0; neighbor < configuration.nNeighbors; ++neighbor) {
          double[] randomVector = IntStream.range(
                  0, genotypeLength) // Extracts component with a Gaussian distribution to have
              // uniformity on the sphere
              .mapToDouble(i -> genotypeBounds.min()
                  + random.nextGaussian() * (genotypeBounds.max() - genotypeBounds.min()))
              .toArray();
          double randomVector_norm = Math.sqrt(Arrays.stream(randomVector)
              .boxed()
              .mapToDouble(element -> element * element)
              .sum());
          double[] neighborGenotype = IntStream.range(0, genotypeLength)
              .mapToDouble(i -> (randomVector[i] / randomVector_norm) * configuration.segmentLength
                  + centralGenotype[i])
              .toArray();
          double[] sampleStep = IntStream.range(0, genotypeLength)
              .mapToDouble(i -> (neighborGenotype[i] - centralGenotype[i]) / (configuration.nSamples - 1))
              .toArray();

          int finalNeighbor = neighbor;
          for (int sample = 1; sample < configuration.nSamples; ++sample) {
            int finalSample = sample;
            executorService.submit(() -> {
              StringBuilder line = new StringBuilder();
              line.append("%s,%s,%d,%d,%d,%.2e,%d,"
                  .formatted(
                      problem.environment,
                      problem.builder,
                      finalPoint,
                      finalNeighbor,
                      finalSample,
                      configuration.segmentLength,
                      genotypeLength));
              double[] sampleGenotype = Arrays.stream(sampleStep)
                  .boxed()
                  .mapToDouble(s -> s * finalSample)
                  .toArray();
              double[] fitnessValues = getFitnessValues(problem, sampleGenotype);
              line.append(Arrays.stream(fitnessValues)
                  .mapToObj(value -> String.format("%.5e", value))
                  .collect(Collectors.joining(",")));
              ps.println(line);
              counterSimulation.getAndIncrement();
            });
          }
        }
      }
    }
    executorService.shutdown();
    boolean terminated = false;
    while (!terminated) {
      try {
        terminated = executorService.awaitTermination(1, TimeUnit.SECONDS);
      } catch (InterruptedException e) {
        // ignore
      }
    }
    updatePrinterExecutor.shutdown();
    System.out.println("Done");
    ps.close();
  }
}
