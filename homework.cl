/*
 *The valid_cell() function checks if the given cell_index is valid by checking
 *if its type is 'f' in type_matrix_cl. If it is, it returns true, otherwise
 * false.
 */
bool valid_cell(int cell_index, __global char *type_matrix_cl) {
  if (type_matrix_cl[cell_index] == 'f')
    return true;
  return false;
}

/*
 *The calculate_temperature() function calculates the temperature of a cell at a
 *given cell_index by summing up all of its neighboring cells' temperatures and
 *dividing it by the number of neighbors. It does this by looping through all 8
 *neighboring cells and checking if they are valid (using valid_cell()) before
 *adding their temperatures to temp_sum.
 */
double calculate_temperature(int cell_index, int X, int Y,
                             __global double *curr_matrix_cl,
                             __global char *type_matrix_cl) {

  int line_index, column_index;
  line_index = cell_index / Y;
  column_index = cell_index - line_index * Y;

  double temp_sum = 0.0;
  int sum_counter = 0;

  for (int i = line_index - 1; i <= line_index + 1; i++) {
    for (int j = column_index - 1; j <= column_index + 1; j++) {
      int temp_index = i * Y + j;
      if (!valid_cell(temp_index, type_matrix_cl))
        continue;
      if (i >= X || i < 0)
        continue;
      if (j >= Y || j < 0)
        continue;
      temp_sum += curr_matrix_cl[temp_index];
      sum_counter++;
    }
  }

  return temp_sum / sum_counter;
}

/*
 *Replaces old value with new one
 */
void assign_value(double new_value, __global double *old_value) {
  *old_value = new_value;
}

/*
 *This code is a kernel function for temperature calculations.
 *- curr_matrix_cl: a global double array that stores the current temperature
 *values
 *- type_matrix_cl: a global char array that stores the type of cell (f for
 *fixed, v for variable)
 *- dim_cl: a global int array that stores the dimensions of the matrix (X and
 *Y)
 *- next_matrix_cl: a global double array that stores the next temperature
 *values
 */
__kernel void temperature_calculations(__global double *curr_matrix_cl,
                                       __global char *type_matrix_cl,
                                       __global int *dim_cl,
                                       __global double *next_matrix_cl) {

  int worker_id = get_global_id(0);
  int workers_count = get_global_size(0);

  int X = dim_cl[0];
  int Y = dim_cl[1];
  int total_size = X * Y;

  // In case the number of worker items exceeds the total size of work, ommit
  if (!(worker_id > total_size)) {

    // How much work each work item has to achieve
    // If total workers
    int workers_work_size = total_size / min(workers_count, total_size);

    // Start cell_index
    int start_i = workers_work_size * worker_id;
    // Stop index
    int stop_i;
    // If it's the last worker, make it work to the end
    if (worker_id == workers_count - 1) {
      stop_i = total_size;
    } else {
      stop_i = workers_work_size * (worker_id + 1);
    }
    // Start the for loop
    for (int cell_index = start_i; cell_index < stop_i; cell_index++) {

      if (!valid_cell(cell_index, type_matrix_cl)) {
        continue;
      }

      assign_value(calculate_temperature(cell_index, X, Y, curr_matrix_cl,
                                         type_matrix_cl),
                   &(next_matrix_cl[cell_index]));
    }
  }
}
