use crate::iris_nn::calc_average_and_variance;

pub fn iris_normalization(data: &[irisdata::Iris; 150]) -> [irisdata::Iris; 150] {
    let (sepal_length_average, sepal_length_variance, _) =
        calc_average_and_variance(&data.iter().map(|d| d.sepal_length).collect::<Vec<f32>>());
    let (sepal_width_average, sepal_width_variance, _) =
        calc_average_and_variance(&data.iter().map(|d| d.sepal_length).collect::<Vec<f32>>());
    let (petal_length_average, petal_length_variance, _) =
        calc_average_and_variance(&data.iter().map(|d| d.petal_length).collect::<Vec<f32>>());
    let (petal_width_average, petal_width_variance, _) =
        calc_average_and_variance(&data.iter().map(|d| d.petal_width).collect::<Vec<f32>>());

    let mut normalized_iris_vec = Vec::<irisdata::Iris>::new();

    for i in 0..data.len() {
        normalized_iris_vec.push(irisdata::Iris {
            sepal_length: (data[i].sepal_length - sepal_length_average)
                / sepal_length_variance.sqrt(),
            sepal_width: (data[i].sepal_width - sepal_width_average)
                / sepal_width_variance.sqrt(),
            petal_length: (data[i].petal_width - petal_length_average)
                / petal_length_variance.sqrt(),
            petal_width: (data[i].petal_width - petal_width_average)
                / petal_width_variance.sqrt(),
            species: data[i].species,
        });
    }

    normalized_iris_vec.try_into().unwrap()
}
