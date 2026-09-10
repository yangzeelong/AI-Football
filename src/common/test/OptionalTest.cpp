#include "../Optional.hpp"
#include <gtest/gtest.h>

namespace {
class Person {
public:
    Person(const std::string& name, int age) : m_name(name), m_age(age) {}
    std::string getName() const { return m_name; }
    int getAge() const { return m_age; }

private:
    std::string m_name;
    int m_age;
};
} // namespace

// 测试用例：成功创建 Person 对象
TEST(OptionalTest, CreatePersonSuccess) {
    auto person = Optional<Person>("Alice", 30);
    ASSERT_TRUE(person.hasValue());
    EXPECT_EQ("Alice", person->getName());
    EXPECT_EQ(30, person->getAge());
}

// 测试用例：未创建 Person 对象
TEST(OptionalTest, CreatePersonFailure) {
    // auto person = Optional<Person>(false); // build error.
    Optional<Person> person = nullOpt;
    ASSERT_FALSE(person.hasValue());
}

TEST(TestOptional, DefaultConstructor) {
    Optional<int> opt;
    EXPECT_FALSE(opt.hasValue());
}

TEST(TestOptional, ValueConstructor) {
    Optional<int> opt(5);
    EXPECT_TRUE(opt.hasValue());
    EXPECT_EQ(*opt, 5);
}

TEST(TestOptional, CopyConstructor) {
    Optional<int> opt1(5);
    Optional<int> opt2(opt1);
    EXPECT_EQ(*opt1, *opt2);

    opt2 = 10;
    EXPECT_EQ(*opt1, 5);
    EXPECT_EQ(*opt2, 10);
}

TEST(TestOptional, MoveConstructor) {
    Optional<int> opt1(5);
    Optional<int> opt2(std::move(opt1));
    EXPECT_FALSE(opt1.hasValue()); // opt1 应该被清空
    EXPECT_EQ(*opt2, 5);
}

TEST(TestOptional, AssignmentOperator) {
    Optional<int> opt1(5);
    Optional<int> opt2;
    opt2 = opt1;
    EXPECT_EQ(*opt1, *opt2);
}

TEST(TestOptional, MoveAssignmentOperator) {
    Optional<int> opt1(5);
    Optional<int> opt2;
    opt2 = std::move(opt1);
    EXPECT_FALSE(opt1.hasValue()); // opt1 应该被清空
    EXPECT_EQ(*opt2, 5);
}

TEST(TestOptional, Reset) {
    Optional<int> opt(5);
    opt.reset(10);
    EXPECT_EQ(*opt, 10);
}

TEST(TestOptional, GetValueRef) {
    Optional<int> opt(5);
    EXPECT_EQ(opt.value(), 5);
    const Optional<int> constOpt(5);
    EXPECT_EQ(constOpt.value(), 5);
}

TEST(TestOptional, OrElse) {
    Optional<int> opt1(5);
    EXPECT_EQ(opt1.orElse(10), 5);
    Optional<int> opt2;
    EXPECT_EQ(opt2.orElse(10), 10);
}

TEST(TestOptional, DereferenceOperator) {
    Optional<int> opt(5);
    EXPECT_EQ(*opt, 5);
}

TEST(TestOptional, MemberAccessOperator) {
    Optional<std::string> opt("hello");
    EXPECT_EQ(opt->size(), 5);
}

TEST(TestOptional, ExceptionWhenEmpty) {
    Optional<int> opt;
    EXPECT_THROW(opt.value(), std::runtime_error);
    EXPECT_THROW(*opt, std::runtime_error);
    EXPECT_THROW(opt.operator->(), std::runtime_error);
}

TEST(TestOptional, TestInitNone) {
    Optional<std::string> opt;
    ASSERT_FALSE(opt.hasValue());
}

TEST(TestOptional, TestLvalueSet) {
#define DATA       "Hello, world"
#define EMPTY_DATA ""

    Optional<std::string> opt;
    const std::string message = DATA;

    ASSERT_FALSE(opt.hasValue());
    opt.reset(message);

    ASSERT_TRUE(opt.hasValue());
    ASSERT_EQ(opt.value(), message);
    ASSERT_EQ(message, DATA);
}

TEST(TestOptional, TestRvalueSet) {
#define DATA       "Hello, world"
#define EMPTY_DATA ""

    Optional<std::string> opt;
    std::string message = DATA;

    ASSERT_FALSE(opt.hasValue());
    opt.reset(std::move(message));

    ASSERT_TRUE(opt.hasValue());
    ASSERT_EQ(opt.value(), DATA);
    ASSERT_EQ(message, EMPTY_DATA);
}

TEST(TestOptional, TestEnum) {
    enum Color { RED, BLUE, YELLOW };

    Optional<Color> opt;

    ASSERT_FALSE(opt.hasValue());
    opt.reset(RED);

    ASSERT_EQ(opt.value(), RED);
}

TEST(TestOptional, TestOrElse) {
    enum Color : int32_t { RED, BLUE, YELLOW };

    Optional<Color> opt;
    ASSERT_FALSE(opt.hasValue());

    ASSERT_EQ(opt.orElse(RED), RED);
    ASSERT_EQ(opt.orElse(BLUE), BLUE);
    ASSERT_EQ(opt.orElse(YELLOW), YELLOW);

    opt.reset(RED);
    ASSERT_EQ(opt.orElse(BLUE), RED);
    ASSERT_EQ(opt.orElse(YELLOW), RED);
}

TEST(TestOptional, TestPointerValue) {
    // Optional<int*> opt; // build error.
    // Optional<std::reference_wrapper<int>> opt2; // build error.
    // Optional<int&> opt3; // build error.
}
