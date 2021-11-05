use plotters::prelude::*;

pub fn graph_results(name: &str, epochs: usize, data: Vec<(f32, f64)>, min: f64, max: f64) {
    let file = format!("{}.svg", name);
    let root = SVGBackend::new(&file, (2000, 1000)).into_drawing_area();

    root.fill(&WHITE).unwrap();

    let root = root.margin(10, 10, 10, 10);

    let y_range = max - min;
    let y_range = (min - 0.1 * y_range)..(max + 0.1 * y_range);

    let mut chart = ChartBuilder::on(&root)
        .caption(name, ("sans-serif", 20).into_font())
        .x_label_area_size(20)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..epochs as f32, y_range)
        .unwrap();

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .y_label_formatter(&|y| format!("{:.3}", y))
        .x_label_formatter(&|x| format!("{}", *x as u64))
        .draw()
        .unwrap();

    let series = LineSeries::new(data.clone(), &BLACK);
    chart.draw_series(series).unwrap();

    chart
        .draw_series(PointSeries::of_element(
            data.clone(),
            5,
            &RED,
            &|c, s, st| {
                return EmptyElement::at(c)
                    + Circle::new((0, 0), s, st.filled())
                    + Text::new(
                        format!("{:.3?}", c),
                        (10, 0),
                        ("sans-serif", 10).into_font(),
                    );
            },
        ))
        .unwrap();
}
