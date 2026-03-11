CC = gcc
FLAGS = -Wall -Wextra -Werror -fsanitize=address -O2 -march=native -mtune=native -ffast-math

OBJS = build/dqn.o build/dnn.o
LIB = build/libdqn.a
LIBDIR = /usr/local/lib
INCDIR = /usr/local/include/dqn

all: build

build/%.o: src/%.c
	mkdir -p build
	$(CC) $(FLAGS) -c $< -o $@

build: $(OBJS)
	ar rcs $(LIB) $(OBJS)

clean:
	rm -rf build

install:
	mkdir -p $(LIBDIR)
	mkdir -p $(INCDIR)
	cp $(LIB) $(LIBDIR)
	cp include/*.h $(INCDIR)

uninstall:
	rm -f $(LIBDIR)/libdqn.a
	rm -rf $(INCDIR)