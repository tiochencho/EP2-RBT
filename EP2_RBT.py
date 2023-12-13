from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt


# Posiciones iniciales y finales 
xi = 0.15  
yi = 0.15  
xf = 0.5  
yf = 0.7   

# Tiempo inicial y final
ti = 0     
tf = 10    

T = 0.001  # Periodo de muestreo
tiempo = np.arange(ti, tf, T)

coeficientes_x = np.polyfit([ti, tf], [xi, xf], 5) 
coeficientes_y = np.polyfit([ti, tf], [yi, yf], 5)  

trayectoria_xd = np.polyval(coeficientes_x, tiempo)
trayectoria_yd = np.polyval(coeficientes_y, tiempo)

# Trayectorias xd(t) y yd(t)
plt.figure(figsize=(8, 6))
plt.plot(tiempo, trayectoria_xd, label='Trayectoria en x', color='purple')
plt.plot(tiempo, trayectoria_yd, label='Trayectoria en y', color='black')
plt.title('Trayectorias Polinomiales en el plano XY')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición (metros)')
plt.legend()
plt.grid(True)
plt.show()

def calcular_xyz(x, y, z, t, coeficientes_x, coeficientes_y, coeficientes_z):
    xd = coeficientes_x[0] * x + coeficientes_x[1] * x * t**2 + coeficientes_x[2] * x * t**3 + coeficientes_x[3] * x * t**4 + coeficientes_x[4] * x * t**5
    yd = coeficientes_y[0] * y + coeficientes_y[1] * y * t**2 + coeficientes_y[2] * y * t**3 + coeficientes_y[3] * y * t**4 + coeficientes_y[4] * y * t**5
    zd = coeficientes_z[0] * z + coeficientes_z[1] * z * t**2 + coeficientes_z[2] * z * t**3 + coeficientes_z[3] * z * t**4 + coeficientes_z[4] * z * t**5
    return xd, yd, zd

# Suponiendo valores arbitrarios para x, y, z, t y los coeficientes para xd, yd, zd
posicion_inicial_x = 3.0
posicion_inicial_y = 2.0
posicion_inicial_z = 4.0
tiempo = 2.0
coeficientes_x = (1.0, 2.0, 0.5, 0.3, 0.2)
coeficientes_y = (0.8, 1.5, 0.6, 0.4, 0.2)
coeficientes_z = (1.2, 2.0, 0.7, 0.5, 0.3)

# Calcular xd, yd y zd utilizando la función calcular_xyz
posicion_xd, posicion_yd, posicion_zd = calcular_xyz(posicion_inicial_x, posicion_inicial_y, posicion_inicial_z, tiempo, coeficientes_x, coeficientes_y, coeficientes_z)

# Mostrar los resultados
print("La posición xd en el tiempo", tiempo, "es:", posicion_xd)
print("La posición yd en el tiempo", tiempo, "es:", posicion_yd)
print("La posición zd en el tiempo", tiempo, "es:", posicion_zd)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Función que define las ecuaciones de cinemática inversa
def cinematica_inversa(q, xd, yd):
    L = 15  # Longitud de los eslabones en centímetros
    eq1 = xd - (L * np.cos(q[0]) + L * np.cos(q[0] + q[1]))
    eq2 = yd - (L * np.sin(q[0]) + L * np.sin(q[0] + q[1]))
    return [eq1, eq2]

# Valores de xd y yd (suponiendo valores aleatorios)
tiempo = np.linspace(0, 10, 100)  # Valores de tiempo
xd_t = np.random.rand(len(tiempo)) * 10  # Valores aleatorios para xd(t)
yd_t = np.random.rand(len(tiempo)) * 10  # Valores aleatorios para yd(t)

# Resolver iterativamente para encontrar q1 y q2 para cada punto en el tiempo
q1_valores = []
q2_valores = []

for i in range(len(tiempo)):
    # Adivinanza inicial para q1 y q2
    q_guess = [0.1, 0.1]  # Suposición inicial de ángulos en radianes
    
    # Resolver las ecuaciones de cinemática inversa para el tiempo t
    sol = root(cinematica_inversa, q_guess, args=(xd_t[i], yd_t[i]))

    if sol.success:
        q1, q2 = np.degrees(sol.x)  # Convertir a grados
        q1_valores.append(q1)
        q2_valores.append(q2)
    else:
        q1_valores.append(np.nan)  # Marcar como NaN si no se encuentra solución
        q2_valores.append(np.nan)  # Marcar como NaN si no se encuentra solución

# Graficar las trayectorias q1(t) y q2(t)
plt.figure(figsize=(8, 6))
plt.plot(tiempo, q1_valores, label='Ángulo q1', color='red')
plt.plot(tiempo, q2_valores, label='Ángulo q2', color='grey')
plt.title('Ángulos en función del tiempo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo (grados)')
plt.legend()
plt.grid(True)
plt.show()
