#include <random>
class Random {
private:
  static inline std::default_random_engine generator{};
public:
  static int random_int_in_interval(int lo,int hi) {
    std::uniform_int_distribution distribution(lo,hi);
    return distribution(generator);
  };
};

int main() {
  int die = Random::random_int_in_interval(1,6);
  return 0;
}

