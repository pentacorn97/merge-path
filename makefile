CC = nvcc
DIR_SRC = ./src
DIR_OBJ = ./obj
DIR_INC = ./inc
FLAG = --allow-unsupported-compiler --std c++14

SRC = $(wildcard $(DIR_SRC)/*.cu)
OBJ_ALL = $(patsubst %.cu, %.obj, $(addprefix $(DIR_OBJ)/, $(notdir $(SRC))))
REBUILDABLE = ./*.exe ./*.exp ./*.lib ./*.obj $(OBJ_ALL)

clean:
	rm -f $(REBUILDABLE)

$(DIR_OBJ)/%.obj : $(DIR_SRC)/%.cu
	$(CC) $(FLAG) -o $@ -c $<

test : $(DIR_OBJ)/test.obj
	$(CC) $(FLAG) -o $@ $^

test_big: $(DIR_OBJ)/test_big.obj
	$(CC) $(FLAG) -o $@ $^

test_all: $(DIR_OBJ)/test_all.obj
	$(CC) $(FLAG) -o $@ $^

test_misc: $(DIR_OBJ)/test_misc.obj
	$(CC) $(FLAG) -o $@ $^