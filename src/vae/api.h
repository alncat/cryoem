void initialise_model_optimizer(int, int, int, float, int);
int train_a_batch(std::vector<float> &data, int part_id);
void optimize_ctf(std::vector<float>& image_data, std::vector<float>& ref_data, float& delta_u, float& delta_v, float& angle, float defocus_res, std::vector<float> Ks, float Q0, int image_size, float angpix, bool do_damping);
