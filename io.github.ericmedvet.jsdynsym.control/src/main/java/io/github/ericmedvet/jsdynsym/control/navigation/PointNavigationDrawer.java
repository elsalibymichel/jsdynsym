/*-
 * ========================LICENSE_START=================================
 * jsdynsym-control
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
package io.github.ericmedvet.jsdynsym.control.navigation;

import io.github.ericmedvet.jsdynsym.control.Simulation;
import io.github.ericmedvet.jsdynsym.control.SimulationOutcomeDrawer;
import io.github.ericmedvet.jsdynsym.control.SingleAgentTask;
import io.github.ericmedvet.jsdynsym.control.geometry.Point;
import io.github.ericmedvet.jviz.core.util.GraphicsUtils;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.awt.geom.Path2D;
import java.util.Arrays;
import java.util.List;
import java.util.SortedMap;
import java.util.stream.Collectors;

public class PointNavigationDrawer
    implements SimulationOutcomeDrawer<SingleAgentTask.Step<double[], double[], PointNavigationEnvironment.State>> {

  private static final int DEFAULT_SIDE_LENGTH = 500;

  private final Configuration configuration;

  public PointNavigationDrawer(Configuration configuration) {
    this.configuration = configuration;
  }

  public record Configuration(
      Color robotColor,
      Color targetColor,
      Color segmentColor,
      Color infoColor,
      double robotThickness,
      double targetThickness,
      double segmentThickness,
      double trajectoryThickness,
      double robotFillAlpha,
      double targetSize,
      double robotDotSize,
      double marginRate) {

    public static final Configuration DEFAULT =
        new Configuration(Color.BLUE, Color.CYAN, Color.DARK_GRAY, Color.BLUE, 2, 2, 3, 1, 0.25, 5, .01, 0.01);
  }

  private static void drawTarget(Graphics2D g, Color c, double th, double l, Point p) {
    g.setStroke(new BasicStroke((float) th));
    g.setColor(c);
    g.draw(new Line2D.Double(p.x() - l / 2d, p.y(), p.x() + l / 2d, p.y()));
    g.draw(new Line2D.Double(p.x(), p.y() - l / 2d, p.x(), p.y() + l / 2d));
  }

  private static void drawTrajectory(Graphics2D g, Color c, double th, List<Point> points) {
    g.setStroke(new BasicStroke((float) th));
    g.setColor(c);
    Path2D path = new Path2D.Double();
    path.moveTo(points.get(0).x(), points.get(0).y());
    points.forEach(p -> path.lineTo(p.x(), p.y()));
    g.draw(path);
  }

  private void drawRobot(Graphics2D g, Color c, double alpha, double th, Point p) {
    g.setStroke(new BasicStroke((float) th));
    Shape shape = new Ellipse2D.Double(
        p.x() - configuration.robotDotSize,
        p.y() - configuration.robotDotSize,
        2d * configuration.robotDotSize,
        2d * configuration.robotDotSize);
    g.setColor(GraphicsUtils.alphaed(c, alpha));
    g.fill(shape);
    g.setColor(c);
    g.draw(shape);
  }

  private static void drawIO(Graphics2D g, Color c, double[] in, double[] out) {
    g.setStroke(new BasicStroke(1f));
    g.setColor(c);
    g.drawString(
        "in:  %s"
            .formatted(
                Arrays.stream(in).mapToObj("%+4.2f"::formatted).collect(Collectors.joining(" "))),
        5,
        5 + g.getFontMetrics().getHeight() * 2);
    g.drawString(
        "out: %s"
            .formatted(
                Arrays.stream(out).mapToObj("%+4.2f"::formatted).collect(Collectors.joining(" "))),
        5,
        5 + g.getFontMetrics().getHeight() * 3);
  }

  @Override
  public void drawSingle(
      Graphics2D g, double t, SingleAgentTask.Step<double[], double[], PointNavigationEnvironment.State> step) {
    Arena arena = step.state().configuration().arena();
    // draw info
    g.setStroke(new BasicStroke(1f));
    g.setColor(configuration.infoColor);
    g.drawString("%.2fs".formatted(t), 5, 5 + g.getFontMetrics().getHeight());
    // set transform
    AffineTransform previousTransform = setTransform(g, arena);
    // draw arena
    g.setStroke(new BasicStroke(
        (float) (configuration.segmentThickness / g.getTransform().getScaleX())));
    g.setColor(configuration.segmentColor);
    arena.segments().forEach(s -> g.draw(new Line2D.Double(s.p1().x(), s.p1().y(), s.p2().x(), s.p2().y())));
    // draw robot
    drawRobot(
        g,
        configuration.robotColor,
        configuration.robotFillAlpha,
        configuration.robotThickness / g.getTransform().getScaleX(),
        step.state().robotPosition());
    // draw target
    drawTarget(
        g,
        configuration.targetColor,
        configuration.targetThickness / g.getTransform().getScaleX(),
        configuration.targetSize / g.getTransform().getScaleX(),
        step.state().targetPosition());
    // restore transformation
    g.setTransform(previousTransform);
    // draw info
    g.setStroke(new BasicStroke(1f));
    g.setColor(configuration.infoColor);
    g.drawString("%.2fs".formatted(t), 5, 5 + g.getFontMetrics().getHeight());
    // draw input and output
    drawIO(g, configuration.infoColor, step.observation(), step.action());
  }

  @Override
  public void drawAll(
      Graphics2D g,
      SortedMap<Double, SingleAgentTask.Step<double[], double[], PointNavigationEnvironment.State>> map) {
    Arena arena = map.values().iterator().next().state().configuration().arena();
    // set transform
    AffineTransform previousTransform = setTransform(g, arena);
    // draw robot and trajectory
    drawTrajectory(
        g,
        configuration.robotColor,
        configuration.trajectoryThickness / g.getTransform().getScaleX(),
        map.values().stream().map(s -> s.state().robotPosition()).toList());
    // draw target and trajectory
    drawTrajectory(
        g,
        configuration.targetColor,
        configuration.trajectoryThickness / g.getTransform().getScaleX(),
        map.values().stream().map(s -> s.state().targetPosition()).toList());
    // restore transformation
    g.setTransform(previousTransform);
  }

  @Override
  public ImageInfo imageInfo(
      Simulation.Outcome<SingleAgentTask.Step<double[], double[], PointNavigationEnvironment.State>> o) {
    Arena arena = o.snapshots()
        .get(o.snapshots().firstKey())
        .state()
        .configuration()
        .arena();
    return new ImageInfo(
        (int)
            (arena.xExtent() > arena.yExtent()
                ? DEFAULT_SIDE_LENGTH * arena.xExtent() / arena.yExtent()
                : DEFAULT_SIDE_LENGTH),
        (int)
            (arena.xExtent() > arena.yExtent()
                ? DEFAULT_SIDE_LENGTH
                : DEFAULT_SIDE_LENGTH * arena.yExtent() / arena.xExtent()));
  }

  private AffineTransform setTransform(Graphics2D g, Arena arena) {
    double cX = g.getClipBounds().x;
    double cY = g.getClipBounds().y;
    double cW = g.getClipBounds().width;
    double cH = g.getClipBounds().getHeight();
    // compute transformation
    double scale = Math.min(
        cW / (1 + 2 * configuration.marginRate) / arena.xExtent(),
        cH / (1 + 2 * configuration.marginRate) / arena.yExtent());
    AffineTransform previousTransform = g.getTransform();
    AffineTransform transform = AffineTransform.getScaleInstance(scale, scale);
    transform.translate(
        (cX / scale + cW / scale - arena.xExtent()) / 2d, (cY / scale + cH / scale - arena.yExtent()) / 2d);
    g.setTransform(transform);
    return previousTransform;
  }
}