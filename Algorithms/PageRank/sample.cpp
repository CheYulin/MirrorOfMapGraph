#include <stdio.h>
class Foo
{
    int Bar;

    public:

    int& GetBar() 
    {
        return Bar;
    }
};

int main(int argc, char** argv)
{
  Foo x;
  int y = x.GetBar();
  y = 5;
  printf("Bar=%d\n", x.GetBar());
  return 0;
}
